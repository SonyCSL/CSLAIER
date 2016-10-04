# -*- coding: utf-8 -*-
from __future__ import print_function
import datetime
import json
import random
import multiprocessing
import threading
import time
import imp
import re
import os
import shutil
import math
from logging import getLogger

import numpy as np
from PIL import Image
import six
import cPickle as pickle
from six.moves import queue

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

from .utils import remove_resume_file

try:
    import tensorflow as tf
except:
    pass

VALIDATION_TIMING = 500  # ORIGINAL 50000

logger = getLogger(__name__)


def _create_trained_model_dir(path, root_output_dir, model_name):
    if path is None:
        path = os.path.join(root_output_dir, model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def _delete_old_models(db_model, pretrained_model):
    pretrained_models = sorted(os.listdir(db_model.trained_model_path), reverse=True)
    for m in pretrained_models:
        if m.startswith('model') and pretrained_model != m:
            try:
                os.remove(os.path.join(db_model.trained_model_path, m))
            except Exception as e:
                logger.exception('Could not remove old models: {0} {1}'
                                 .format(os.path.join(db_model.trained_model_path, m), e))
                raise e


def _backup_pretrained_model(db_model, pretrained_model):
    try:
        shutil.copyfile(os.path.join(db_model.trained_model_path, pretrained_model),
                        os.path.join(db_model.trained_model_path,
                                     'previous_' + pretrained_model))
    except Exception as e:
        logger.exception('Could not copy {0} to {1}. {2}'
                         .format(os.path.join(db_model.trained_model_path, pretrained_model),
                                 os.path.join(db_model.trained_model_path,
                                              'previous_' + pretrained_model), e))
        raise e


def _post_process(db_model, pretrained_model):
    # post-processing
    db_model.is_trained = 2
    db_model.pid = None
    db_model.gpu = None
    db_model.update_and_commit()
    if os.path.exists(os.path.join(db_model.trained_model_path, 'previous_' + pretrained_model)):
        # delete backup file
        try:
            os.remove(os.path.join(db_model.trained_model_path, 'previous_' + pretrained_model))
        except Exception as e:
            logger.exception('Could not delete backuped model: {0} {1}'
                             .format(os.path.join(db_model.trained_model_path,
                                                  'previous_' + pretrained_model), e))
            raise e
    # delete prepared images
    for f in os.listdir(db_model.prepared_file_path):
        (head, ext) = os.path.splitext(f)
        ext = ext.lower()
        if ext in ['.jpg', '.jpeg', '.gif', '.png', '.tfrecord']:
            try:
                os.remove(os.path.join(db_model.prepared_file_path, f))
            except Exception as e:
                logger.exception('Could not remove prepared file: {0} {1}'
                                 .format(os.path.join(db_model.prepared_file_path, f), e))
                raise e


def load_image_list(path):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((pair[0], np.int32(pair[1])))
    return tuples


def read_image(path, model_insize, mean_image, center=False, flip=False, original_size=256):
    cropwidth = original_size - model_insize
    image = np.asarray(Image.open(path))
    if len(image.shape) == 3:
        image = image.transpose(2, 0, 1)
    else:
        zeros = np.zeros((original_size, original_size))
        image = np.array([image, zeros, zeros])
    if center or model_insize == original_size:
        top = left = cropwidth / 2
    else:
        top = random.randint(0, cropwidth - 1)
        left = random.randint(0, cropwidth - 1)
    bottom = model_insize + top
    right = model_insize + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    image -= mean_image[:, top:bottom, left:right]
    image /= 255
    if flip and random.randint(0, 1) == 0:
        return np.fliplr(image)
    else:
        return image


# 状態の再現に必要でかつSQLiteに格納されていないものはこのクラスが持ち、JSON化します。
class TrainingEpoch(object):
    def __init__(self, number_of_epoch, batch_size, permutation=None, avoid_flipping=False, pretrained_model=None):
        super(TrainingEpoch, self).__init__()
        self.epoch = number_of_epoch
        self.permutation = permutation
        self.avoid_flipping = avoid_flipping
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.number_of_before_epoch_train = 0

    # self == 'epoch' == Trueになります。train_loop内の取り回しのため。
    def __eq__(self, other):
        return other == 'epoch'

    # 済んだ学習のidxを削っていく(Epochを跨いでいる分を加味)
    def batch_updated(self):
        self.permutation = self.permutation[(self.batch_size - self.number_of_before_epoch_train):]
        self.number_of_before_epoch_train = 0

    def next_number_of_before_epoch_train(self):
        return len(self.permutation) + self.number_of_before_epoch_train

    def serialize(self, fp):
        json.dump({
            'epoch': self.epoch,
            'batch_size': self.batch_size,
            'permutation': list(self.permutation),
            'avoid_flipping': self.avoid_flipping,
            'pretrained_model': self.pretrained_model
        }, fp)

    @staticmethod
    def deserialize(json_file):
        d = json.load(open(json_file))
        return TrainingEpoch(
            d['epoch'],
            d['batch_size'],
            np.array(d['permutation']),
            d['avoid_flipping'],
            d['pretrained_model'])


def feed_data(train_list, val_list, mean_image, batchsize, val_batchsize,
              model, loaderjob, epoch, optimizer, data_q, avoid_flipping, resume_perm=None, resume_epoch=1):
    denominator = 1000 if len(train_list) > 1000 else len(train_list)
    i = 0
    count = 0
    x_batch = np.ndarray((batchsize, 3, model.insize, model.insize), dtype=np.float32)
    y_batch = np.ndarray((batchsize,), dtype=np.int32)
    val_x_batch = np.ndarray((val_batchsize, 3, model.insize, model.insize), dtype=np.float32)
    val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)

    batch_pool = [None] * batchsize
    val_batch_pool = [None] * val_batchsize
    pool = multiprocessing.Pool(loaderjob)
    data_q.put('train')
    use_flip = True

    if avoid_flipping:
        use_flip = False

    for epoch in six.moves.range(resume_epoch, 1 + epoch):
        logger.info('epoch: {0}'.format(epoch))
        logger.info('learning rate: {0}'.format(optimizer.lr))
        if resume_perm is not None and resume_epoch == epoch:
            perm = resume_perm
        else:
            perm = np.random.permutation(len(train_list))
        data_q.put(TrainingEpoch(epoch - 1, batchsize, perm, avoid_flipping))
        for idx in perm:
            path, label = train_list[idx]
            batch_pool[i] = pool.apply_async(read_image, (path, model.insize, mean_image,
                                                          False, use_flip))
            y_batch[i] = label
            i += 1

            if i == batchsize:
                for j, x in enumerate(batch_pool):
                    x_batch[j] = x.get()
                data_q.put((x_batch.copy(), y_batch.copy(), epoch))
                i = 0

            count += 1
            if count % denominator == 0:
                data_q.put('val')
                j = 0
                for path, label in val_list:
                    val_batch_pool[j] = pool.apply_async(read_image,
                                                         (path, model.insize,
                                                          mean_image, True, False))
                    val_y_batch[j] = label
                    j += 1

                    if j == val_batchsize:
                        for k, x in enumerate(val_batch_pool):
                            val_x_batch[k] = x.get()
                        data_q.put((val_x_batch.copy(), val_y_batch.copy(), epoch))
                        j = 0
                data_q.put('train')

        optimizer.lr *= 0.97
    pool.close()
    pool.join()
    data_q.put('end')
    return


def log_result(batchsize, val_batchsize, log_file, log_html, train_log, res_q, resume=False):
    print(log_file)
    if resume:
        fH = open(log_html, 'a')
        print(log_file)
        # ログをちゃんと連番にするためにログの最後の行のcountをとる
        with open(log_file) as fp:
            row = '0'
            # 一行目を捨てる(header)
            fp.next()
            for row in filter(lambda line: line.strip(), fp):
                pass
            # 最後の行を取得
            train_count = int(row.split('\t')[0])
        f = open(log_file, 'a')
    else:
        fH = open(log_html, 'w')
        f = open(log_file, 'w')
        f.write("count\tepoch\taccuracy\tloss\taccuracy(val)\tloss(val)\n")
        fH.flush()
        f.flush()
        train_count = 0
    train_cur_loss = 0
    train_cur_accuracy = 0
    begin_at = time.time()
    val_begin_at = None
    while True:
        result = res_q.get()
        if result == 'end':
            break
        elif result == 'train':
            train = True
            if val_begin_at is not None:
                begin_at += time.time() - val_begin_at
                val_begin_at = None
            continue
        elif result == 'val':
            train = False
            val_count = val_loss = val_accuracy = 0
            val_begin_at = time.time()
            continue
        loss, accuracy, epoch = result
        if train:
            train_count += 1
            duration = time.time() - begin_at
            throughput = train_count * batchsize / duration
            with open(train_log, 'a') as fp:
                fp.write(
                    json.dumps({'train': train_count,
                                'updates': train_count * batchsize,
                                'time': '{} ({} images/sec)'.format(datetime.timedelta(seconds=duration), throughput),
                                '[TIME]': '{},{}'.format(epoch, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                })
                )
                fp.write('\n')
            fH.write(
                'train {} updates ({} samples) time: {} ({} images/sec)<br>'
                    .format(train_count, train_count * batchsize,
                            datetime.timedelta(seconds=duration), throughput))
            fH.write("[TIME]{},{}<br>"
                     .format(epoch, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            fH.flush()
            f.write(str(train_count) + "\t" + str(epoch) + "\t" + str(accuracy) + "\t" + str(loss)
                    + "\t\t\n")
            f.flush()
            train_cur_loss += loss
            train_cur_accuracy += accuracy
            if train_count % 1000 == 0:
                mean_loss = train_cur_loss / 1000
                mean_error = 1 - train_cur_accuracy / 10000
                fH.write("<strong>"
                         + json.dumps({
                    'type': 'train',
                    'iteration': train_count,
                    'error': mean_error,
                    'loss': mean_loss})
                         + "</strong><br>")
                fH.flush()
                train_cur_loss = 0
                train_cur_accuracy = 0
        else:
            val_count += val_batchsize
            duration = time.time() - val_begin_at
            throughput = val_count / duration
            fH.write(
                'val {} batches ({} samples) time: {} ({} images/sec)'
                    .format(val_count / val_batchsize, val_count,
                            datetime.timedelta(seconds=duration), throughput)
            )
            fH.flush()
            val_loss += loss
            val_accuracy += accuracy
            if val_count == VALIDATION_TIMING:
                mean_loss = val_loss * val_batchsize / VALIDATION_TIMING
                mean_accuracy = val_accuracy * val_batchsize / VALIDATION_TIMING
                fH.write("<strong>"
                         + json.dumps({
                    'type': 'val',
                    'iteration': train_count,
                    'error': (1 - mean_accuracy),
                    'loss': mean_loss})
                         + "</strong><br>")
                fH.flush()
                f.write(str(train_count) + "\t" + str(epoch) + "\t\t\t"
                        + str(mean_accuracy) + "\t" + str(mean_loss) + "\n")
                train_count += 1
                f.flush()
    f.close()
    fH.close()


def train_loop(model, output_dir, xp, optimizer, res_q, data_q, pretrained_model, interrupt_event, interruptable_event):
    graph_generated = False
    training_epoch = None
    while True:
        if interrupt_event.is_set():
            resume_path = os.path.join(output_dir, 'resume')
            os.mkdir(resume_path)
            serializers.save_npz(os.path.join(resume_path, 'resume.model'), model)
            serializers.save_npz(os.path.join(resume_path, 'resume.state'), optimizer)
            training_epoch.pretrained_model = pretrained_model
            training_epoch.serialize(open(os.path.join(resume_path, 'resume.json'), 'w'))
            interruptable_event.set()
            while True:
                time.sleep(1)

        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()

        if inp == 'end':
            res_q.put('end')
            break
        elif inp == 'epoch':
            if training_epoch:
                inp.number_of_before_epoch_train = training_epoch.next_number_of_before_epoch_train()
            training_epoch = inp
            continue
        elif inp == 'train':
            res_q.put('train')
            model.train = True
            continue
        elif inp == 'val':
            res_q.put('val')
            model.train = False
            continue
        volatile = 'off' if model.train else 'on'
        x = chainer.Variable(xp.asarray(inp[0]), volatile=volatile)
        t = chainer.Variable(xp.asarray(inp[1]), volatile=volatile)
        if model.train:
            optimizer.update(model, x, t)
            training_epoch.batch_updated()
            if not graph_generated:
                with open(output_dir + os.sep + 'graph.dot', 'w') as o:
                    o.write(computational_graph.build_computational_graph((model.loss,)).dump())
                graph_generated = True
        else:
            model(x, t)

        serializers.save_hdf5(os.path.join(output_dir, 'model{:04d}'.format(inp[2])), model)

        res_q.put((float(model.loss.data), float(model.accuracy.data), inp[2]))
        del x, t


def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol, [dir_name])
    return imp.load_module(symbol, file, path, description)


def do_train_by_chainer(
        db_model,
        root_output_dir,
        val_batchsize=250,
        loaderjob=20,
        pretrained_model="",
        avoid_flipping=False,
        interrupt_event=None,
        interruptable_event=None,
):
    logger.info('Start imagenet train. model_id: {0} gpu: {1}, pretrained_model: {2}'
                .format(db_model.id, db_model.gpu, pretrained_model))
    # start initialization
    if db_model.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if db_model.gpu >= 0 else np

    train_list = load_image_list(os.path.join(db_model.prepared_file_path, 'train.txt'))
    val_list = load_image_list(os.path.join(db_model.prepared_file_path, 'test.txt'))
    mean_image = pickle.load(open(os.path.join(db_model.prepared_file_path, 'mean.npy'), 'rb'))

    # @see http://qiita.com/progrommer/items/abd2276f314792c359da
    (model_dir, model_name) = os.path.split(db_model.network_path)
    model_name = re.sub(r"\.py$", "", model_name)
    model_module = load_module(model_dir, model_name)
    model = model_module.Network()

    # create directory for saving trained models
    db_model.trained_model_path = _create_trained_model_dir(db_model.trained_model_path,
                                                            root_output_dir, model_name)

    # Load pretrained model
    if pretrained_model is not None and pretrained_model.find("model") > -1:
        logger.info("load pretrained model : "
                    + os.path.join(db_model.trained_model_path, pretrained_model))
        serializers.load_hdf5(os.path.join(db_model.trained_model_path, pretrained_model), model)
        _backup_pretrained_model(db_model, pretrained_model)
        _delete_old_models(db_model, pretrained_model)

    # delete layer visualization cache
    for f in os.listdir(db_model.trained_model_path):
        if os.path.isdir(os.path.join(db_model.trained_model_path, f)):
            try:
                shutil.rmtree(os.path.join(db_model.trained_model_path, f))
            except Exception as e:
                logger.exception('Could not remove visualization cache. {0}'.format(e))
                raise e

    if db_model.gpu >= 0:
        cuda.get_device(db_model.gpu).use()
        model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    data_q = queue.Queue(maxsize=1)
    res_q = queue.Queue()

    db_model.is_trained = 1
    db_model.update_and_commit()

    remove_resume_file(db_model.trained_model_path)

    # Invoke threads
    feeder = threading.Thread(
        target=feed_data,
        args=(
            train_list,
            val_list,
            mean_image,
            db_model.batchsize,
            val_batchsize,
            model,
            loaderjob,
            db_model.epoch,
            optimizer,
            data_q,
            avoid_flipping,
        )
    )
    feeder.daemon = True
    feeder.start()
    train_logger = threading.Thread(
        target=log_result,
        args=(
            db_model.batchsize,
            val_batchsize,
            os.path.join(db_model.trained_model_path, 'line_graph.tsv'),
            os.path.join(db_model.trained_model_path, 'log.html'),
            os.path.join(db_model.trained_model_path, 'train.log'),
            res_q
        )
    )
    train_logger.daemon = True
    train_logger.start()
    train_loop(
        model,
        db_model.trained_model_path,
        xp,
        optimizer,
        res_q,
        data_q,
        pretrained_model,
        interrupt_event,
        interruptable_event
    )
    feeder.join()
    train_logger.join()

    # post-processing
    _post_process(db_model, pretrained_model)
    interrupt_event.clear()
    logger.info('Finish imagenet train. model_id: {0}'.format(db_model.id))


def resume_train_by_chainer(
        db_model,
        root_output_dir,
        val_batchsize=250,
        loaderjob=20,
        interrupt_event=None,
        interruptable_event=None,
):
    logger.info('resume last imagenet train. model_id: {0} gpu: {1}'
                .format(db_model.id, db_model.gpu))
    # start initialization
    if db_model.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if db_model.gpu >= 0 else np

    train_list = load_image_list(os.path.join(db_model.prepared_file_path, 'train.txt'))
    val_list = load_image_list(os.path.join(db_model.prepared_file_path, 'test.txt'))
    mean_image = pickle.load(open(os.path.join(db_model.prepared_file_path, 'mean.npy'), 'rb'))

    # @see http://qiita.com/progrommer/items/abd2276f314792c359da
    (model_dir, model_name) = os.path.split(db_model.network_path)
    model_name = re.sub(r"\.py$", "", model_name)
    model_module = load_module(model_dir, model_name)
    model = model_module.Network()

    # create directory for saving trained models
    db_model.trained_model_path = _create_trained_model_dir(db_model.trained_model_path,
                                                            root_output_dir, model_name)

    if db_model.gpu >= 0:
        cuda.get_device(db_model.gpu).use()
        model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    resume_path = os.path.join(db_model.trained_model_path, 'resume')
    resume_json = os.path.join(resume_path, 'resume.json')

    resume_epoch = TrainingEpoch.deserialize(resume_json)

    # load resume data.
    resume_state = os.path.join(resume_path, 'resume.state')
    resume_model = os.path.join(resume_path, 'resume.model')
    logger.info("Load optimizer state from : {}"
                .format(resume_state))
    serializers.load_npz(resume_model, model)
    serializers.load_npz(resume_state, optimizer)
    remove_resume_file(db_model.trained_model_path)

    data_q = queue.Queue(maxsize=1)
    res_q = queue.Queue()

    db_model.is_trained = 1
    db_model.update_and_commit()

    # Invoke threads
    feeder = threading.Thread(
        target=feed_data,
        args=(
            train_list,
            val_list,
            mean_image,
            db_model.batchsize,
            val_batchsize,
            model,
            loaderjob,
            db_model.epoch,
            optimizer,
            data_q,
            resume_epoch.avoid_flipping,
            resume_epoch.permutation,
            resume_epoch.epoch
        )
    )
    feeder.daemon = True
    feeder.start()
    train_logger = threading.Thread(
        target=log_result,
        args=(
            db_model.batchsize,
            val_batchsize,
            os.path.join(db_model.trained_model_path, 'line_graph.tsv'),
            os.path.join(db_model.trained_model_path, 'log.html'),
            os.path.join(db_model.trained_model_path, 'train.log'),
            res_q,
            True
        )
    )
    train_logger.daemon = True
    train_logger.start()
    train_loop(
        model,
        db_model.trained_model_path,
        xp,
        optimizer,
        res_q,
        data_q,
        resume_epoch.pretrained_model,
        interrupt_event,
        interruptable_event
    )
    feeder.join()
    train_logger.join()

    # post-processing
    _post_process(db_model, resume_epoch.pretrained_model)
    interrupt_event.clear()
    logger.info('Finish imagenet train. model_id: {0}'.format(db_model.id))


def _read_and_decode(filename_queue, avoid_flipping):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [128, 128, 3])
    if not avoid_flipping:
        image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255
    label = tf.cast(features['label'], tf.int32)
    return image, label


def _extract_tfrecord(files, batchsize, num_epochs=None, avoid_flipping=True, use_shuffle=False):
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
    image, label = _read_and_decode(filename_queue, avoid_flipping)
    if use_shuffle:
        images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                       batch_size=batchsize,
                                                       num_threads=8,
                                                       capacity=1000 + 3 * batchsize,
                                                       min_after_dequeue=1000)
    else:
        images, sparse_labels = tf.train.batch([image, label],
                                               batch_size=batchsize,
                                               num_threads=8,
                                               capacity=1000 + 3 * batchsize)

    return images, sparse_labels


def do_train_by_tensorflow(
        db_model,
        output_dir_root,
        val_batchsize,
        pretrained_model,
        train_image_num,
        val_image_num,
        avoid_flipping,
        resume,
        interrupt_event,
        interruptable_event,
):
    logger.info('Start imagenet train. model_id: {}, pretrained_model: {}'
                .format(db_model.id, pretrained_model))

    # load model
    (model_dir, model_name) = os.path.split(db_model.network_path)
    model_name = re.sub(r"\.py$", "", model_name)
    model = load_module(model_dir, model_name)

    db_model.trained_model_path = _create_trained_model_dir(db_model.trained_model_path,
                                                            output_dir_root, model_name)

    db_model.is_trained = 1
    db_model.update_and_commit()

    if db_model.gpu > -1:
        device = '/gpu:' + str(db_model.gpu)
    else:
        device = '/cpu:0'
    train_images, train_sparse_labels = _extract_tfrecord([os.path.join(db_model.prepared_file_path,
                                                                        'train.tfrecord')],
                                                          val_batchsize, num_epochs=db_model.epoch,
                                                          use_shuffle=True)
    val_images, val_sparse_labels = _extract_tfrecord([os.path.join(db_model.prepared_file_path,
                                                                    'test.tfrecord')],
                                                      db_model.batchsize)

    images_placeholder = tf.placeholder(tf.float32)
    labels_placeholder = tf.placeholder(tf.int32)
    keep_prob = tf.placeholder(tf.float32)
    with tf.device(device):
        logits = model.inference(images_placeholder, keep_prob)
        loss_value = model.loss(logits, labels_placeholder)
        train_op = model.training(loss_value, 1e-4)
    acc = model.accuracy(logits, labels_placeholder)

    first_and_last_saver = tf.train.Saver(max_to_keep=2)
    saver = tf.train.Saver(max_to_keep=50)

    if resume:
        mode = 'a'
    else:
        mode = 'w'
    with open(os.path.join(db_model.trained_model_path, 'line_graph.tsv'), mode) as line_graph, \
            open(os.path.join(db_model.trained_model_path, 'log.html'), mode) as log_file:
        log_file.write('train: {} images, val: {} images, epoch: {}<br>'
                       .format(train_image_num, val_image_num, db_model.epoch))
        log_file.flush()
        if not resume:
            line_graph.write("count\tepoch\taccuracy\tloss\taccuracy(val)\tloss(val)\n")
        line_graph.flush()
        with tf.Session() as sess:
            # Load pretrained model
            if pretrained_model is not None and pretrained_model.find("model") > -1:
                logger.info("load pretrained model : "
                            + os.path.join(db_model.trained_model_path, pretrained_model))
                saver.restore(sess, os.path.join(db_model.trained_model_path, pretrained_model))
                _backup_pretrained_model(db_model, pretrained_model)
                _delete_old_models(db_model, pretrained_model)
            resume_path = os.path.join(db_model.trained_model_path, 'resume')
            if resume:
                resume_data = json.load(open(os.path.join(resume_path, 'resume.json')))
                pretrained_model = resume_data.get('pretrained_model', '')
                saver.restore(sess, resume_data['saved_path'])
            else:
                resume_data = {}
                sess.run(tf.initialize_all_variables())

            remove_resume_file(db_model.trained_model_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                step = resume_data.get('step', 0)
                prev_epoch = resume_data.get('prev_epoch', None)
                begin_at = time.time() - resume_data.get('duration', 0)
                train_cur_loss = resume_data.get('train_cur_loss', 0)
                train_cur_accuracy = resume_data.get('train_cur_accuracy', 0)

                while not coord.should_stop():
                    if interrupt_event.is_set():
                        os.mkdir(resume_path)
                        data = {
                            'pretrained_model': pretrained_model,
                            'step': step,
                            'prev_epoch': prev_epoch,
                            'duration': time.time() - begin_at,
                            'train_cur_loss': train_cur_loss,
                            'train_cur_accuracy': train_cur_accuracy,
                            'epoch': current_epoch
                        }
                        saved_path = saver.save(sess, os.path.join(resume_path, 'resume.ckpt'))
                        data['saved_path'] = saved_path
                        json.dump(data, open(os.path.join(resume_path, 'resume.json'), 'w'))
                        interruptable_event.set()
                        while True:
                            time.sleep(0.5)

                    images, sparse_labels = sess.run([train_images, train_sparse_labels])
                    _, train_loss, train_acc = sess.run([train_op, loss_value, acc],
                                                        feed_dict={
                                                            images_placeholder: images,
                                                            labels_placeholder: sparse_labels,
                                                            keep_prob: 0.5})

                    train_cur_loss += train_loss
                    train_cur_accuracy += train_acc

                    current_epoch = int(math.floor(step * db_model.batchsize / train_image_num))

                    line_graph.write('{}\t{}\t{}\t{}\t\t\n'
                                     .format(step, current_epoch, train_acc, train_loss))
                    line_graph.flush()

                    if step % 100 == 0 and step != 0:
                        images, sparse_labels = sess.run([val_images, val_sparse_labels])
                        val_loss_result, val_acc_result = sess.run([loss_value, acc],
                                                                   feed_dict={
                                                                       images_placeholder: images,
                                                                       labels_placeholder: sparse_labels,
                                                                       keep_prob: 1.0})
                        line_graph.write('{}\t{}\t\t\t{}\t{}\n'
                                         .format(step, current_epoch, val_acc_result, val_loss_result))
                        line_graph.flush()
                    if step % 50 == 0 and step != 0:
                        duration = time.time() - begin_at
                        throughput = step * db_model.batchsize / duration
                        log_file.write('train {} updates ({} samples) time: {} ({} images/sec)<br>'
                                       .format(step, step * db_model.batchsize,
                                               datetime.timedelta(seconds=duration), throughput))
                        log_file.write("[TIME]{},{}<br>"
                                       .format(current_epoch,
                                               datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                        log_file.flush()
                        with open(os.path.join(db_model.train_log_path), 'a') as train_log:
                            train_log.write(json.dumps({
                                'train': step,
                                'updates': step * db_model.batchsize,
                                'time': '{} ({} images/sec)'.format(datetime.timedelta(seconds=duration), throughput),
                                '[TIME]': '{},{}'.format(current_epoch,
                                                         datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            }))
                            train_log.write('/n')
                        if step % 1000 == 0:
                            mean_loss = train_cur_loss / 1000
                            mean_error = 1 - train_cur_accuracy / 1000
                            log_file.write('<strong>{}</strong><br>'.format(json.dumps({
                                'type': 'train',
                                'iteration': step,
                                'error': mean_error,
                                'loss': mean_loss
                            })))
                            log_file.flush()
                            train_cur_loss = 0
                            train_cur_accuracy = 0

                    # save trained result
                    if prev_epoch != current_epoch:
                        # epoch updated
                        if current_epoch == 1:
                            first_and_last_saver.save(sess,
                                                      os.path.join(db_model.trained_model_path, 'model'),
                                                      global_step=current_epoch)
                        if current_epoch <= 100:
                            if current_epoch % 10 == 0:
                                saver.save(sess,
                                           os.path.join(db_model.trained_model_path, 'model'),
                                           global_step=current_epoch)
                        else:
                            if current_epoch % 50 == 0:
                                saver.save(sess,
                                           os.path.join(db_model.trained_model_path, 'model'),
                                           global_step=current_epoch)

                    step += 1
                    prev_epoch = current_epoch
            except tf.errors.OutOfRangeError as e:
                logger.info('Epoch limit reached.')
            finally:
                first_and_last_saver.save(sess,
                                          os.path.join(db_model.trained_model_path, 'model'),
                                          global_step=db_model.epoch)
                coord.request_stop()

            coord.join(threads)

    # post-processing
    _post_process(db_model, pretrained_model)
    interrupt_event.clear()
    logger.info('Finish imagenet train. model_id: {0}'.format(db_model.id))
