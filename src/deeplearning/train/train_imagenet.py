# -*- coding: utf-8 -*-
from __future__ import print_function
import datetime
import json
import random
import multiprocessing
import sys
import threading
import time
import imp
import re
import os
import shutil
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

try:
    import tensorflow as tf
except:
    pass

VALIDATION_TIMING = 500 # ORIGINAL 50000

logger = getLogger(__name__)

def _create_trained_model_dir(path, root_output_dir, model_name):
    if path is None:
        path = os.path.join(root_output_dir, model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def _post_process(db_model, pretrained_model):
    # post-processing
    db_model.is_trained = 2
    db_model.pid = None
    db_model.update_and_commit()
    if os.path.exists(os.path.join(db_model.trained_model_path,  'previous_' + pretrained_model)):
        #delete backup file
        try:
            os.remove(os.path.join(db_model.trained_model_path, 'previous_' + pretrained_model))
        except Exception as e:
            logger.exception('Could not delete backuped model: {0} {1}'.format(os.path.join(db_model.trained_model_path, 'previous_' + pretrained_model), e))
            raise e
    # delete prepared images
    for f in os.listdir(db_model.prepared_file_path):
        (head, ext) = os.path.splitext(f)
        ext = ext.lower()
        if ext in ['.jpg', '.jpeg', '.gif', '.png']:
            try:
                os.remove(os.path.join(db_model.prepared_file_path, f))
            except Exception as e:
                logger.exception('Could not remove prepared file: {0} {1}'.format(os.path.join(db_model.prepared_file_path, f), e))
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
        zeros = np.zeros((original_size,original_size))
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

def feed_data(train_list, val_list, mean_image, batchsize, val_batchsize, model, loaderjob, epoch, optimizer, data_q, avoid_flipping):
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

    if avoid_flipping == 1:
        use_flip = False

    for epoch in six.moves.range(1, 1 + epoch):
        logger.info('epoch: {0}'.format(epoch))
        logger.info('learning rate: {0}'.format(optimizer.lr))
        perm = np.random.permutation(len(train_list))
        for idx in perm:
            path, label = train_list[idx]
            batch_pool[i] = pool.apply_async(read_image, (path, model.insize, mean_image, False, use_flip))
            y_batch[i] = label
            i += 1

            if i == batchsize:
                for j, x in enumerate(batch_pool):
                    x_batch[j] = x.get()
                data_q.put((x_batch.copy(), y_batch.copy(), epoch))
                i= 0

            count += 1
            if count % denominator == 0:
                data_q.put('val')
                j = 0
                for path, label in val_list:
                    val_batch_pool[j] = pool.apply_async(read_image, (path, model.insize, mean_image, True, False))
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

def log_result(batchsize, val_batchsize, log_file,log_html, res_q):
    fH = open(log_html, 'w')
    fH.flush()

    f = open(log_file, 'w')
    f.write("count\tepoch\taccuracy\tloss\taccuracy(val)\tloss(val)\n")
    f.flush()
    count = 0
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
            fH.write(
                '\rtrain {} updates ({} samples) time: {} ({} images/sec)<br>'
                .format(train_count, train_count * batchsize, datetime.timedelta(seconds=duration), throughput))
            fH.write("[TIME]{},{}<br>".format(epoch,datetime.datetime.now().strftime( '%Y-%m-%d %H:%M:%S' )))
            fH.flush()
            f.write(str(count) + "\t" + str(epoch) + "\t" + str(accuracy) + "\t" + str(loss) + "\t\t\n")
            f.flush()
            count += 1
            train_cur_loss += loss
            train_cur_accuracy += accuracy
            if train_count % 1000 == 0:
                mean_loss = train_cur_loss / 1000
                mean_error = 1 - train_cur_accuracy / 10000
                fH.write("<strong>"+json.dumps({'type': 'train', 'iteration': train_count, 'error': mean_error, 'loss': mean_loss})+"</strong><br>")
                fH.flush()
                sys.stdout.flush()
                train_cur_loss = 0
                train_cur_accuracy = 0
        else:
            val_count += val_batchsize
            duration = time.time() - val_begin_at
            throughput = val_count / duration
            fH.write(
                '\rval {} batches ({} samples) time: {} ({} images/sec)'
                .format(val_count / val_batchsize, val_count, datetime.timedelta(seconds=duration), throughput)
            )
            fH.flush()
            val_loss += loss
            val_accuracy += accuracy
            if val_count == VALIDATION_TIMING:
                mean_loss = val_loss * val_batchsize / VALIDATION_TIMING
                mean_accuracy = val_accuracy * val_batchsize / VALIDATION_TIMING
                fH.write("<strong>"+json.dumps({'type': 'val', 'iteration': train_count, 'error': (1 - mean_accuracy), 'loss': mean_loss})+"</strong><br>")
                fH.flush()
                f.write(str(count) + "\t" + str(epoch) + "\t\t\t" + str(mean_accuracy) + "\t" + str(mean_loss) + "\n")
                count += 1
                f.flush()
                sys.stdout.flush()
    f.close()
    fH.close()

def train_loop(model, output_dir, xp, optimizer, res_q, data_q):
    graph_generated = False
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':
            res_q.put('end')
            break
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
            if not graph_generated:
                with open(output_dir + os.sep + 'graph.dot', 'w') as o:
                    o.write(computational_graph.build_computational_graph((model.loss,)).dump())
                graph_generated = True
        else:
            model(x, t)

        serializers.save_hdf5(output_dir + os.sep + 'model%04d'%inp[2], model)
        #serializers.save_hdf5(output_dir + os.sep + 'optimizer%04d'%inp[2], optimizer)
        res_q.put((float(model.loss.data), float(model.accuracy.data), inp[2]))
        del x, t

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol,[dir_name])
    return imp.load_module(symbol, file, path, description)

def do_train_by_chainer(
    db_model,
    root_output_dir,
    batchsize=32,
    val_batchsize=250,
    gpu=-1,
    loaderjob=20,
    pretrained_model=""
):
    logger.info('Start imagenet train. model_id: {0} gpu: {1}, pretrained_model: {2}'.format(db_model.id, gpu, pretrained_model))
    # start initialization
    if gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu >= 0 else np

    train_list = load_image_list(os.path.join(db_model.prepared_file_path, 'train.txt'))
    val_list   = load_image_list(os.path.join(db_model.prepared_file_path, 'test.txt'))
    mean_image = pickle.load(open(os.path.join(db_model.prepared_file_path, 'mean.npy'), 'rb'))

    # @see http://qiita.com/progrommer/items/abd2276f314792c359da
    (model_dir, model_name) = os.path.split(db_model.network_path)
    model_name = re.sub(r"\.py$", "", model_name)
    model_module = load_module(model_dir, model_name)
    model = model_module.Network()

    # create directory for saving trained models
    db_model.trained_model_path = _create_trained_model_dir(db_model.trained_model_path, root_output_dir, model_name)

    # Load pretrained model
    if pretrained_model is not None and pretrained_model.find("model") > -1:
        logger.info("load pretrained model : "+db_model.trained_model_path + os.sep +pretrained_model)
        serializers.load_hdf5(os.path.join(db_model.trained_model_path,pretrained_model), model)
        # delete old models
        try:
            shutil.copyfile(os.path.join(db_model.trained_model_path, pretrained_model), os.path.join(db_model.trained_model_path, 'previous_' + pretrained_model))
        except Exception as e:
            logger.exception('Could not copy {0} to {1}. {2}'.format(os.path.join(db_model.trained_model_path, pretrained_model), os.path.join(db_model.trained_model_path, 'previous_' + pretrained_model), e))
            raise e
        pretrained_models = sorted(os.listdir(db_model.trained_model_path), reverse=True)
        for m in pretrained_models:
            if m.startswith('model') and pretrained_model != m:
                try:
                    os.remove(os.path.join(db_model.trained_model_path, m))
                except Exception as e:
                    logger.exception('Could not remove old models: {0} {1}'.format(os.path.join(db_model.trained_model_path, m), e))
                    raise e

    # delete layer visualization cache
    for f in os.listdir(db_model.trained_model_path):
        if os.path.isdir(os.path.join(db_model.trained_model_path, f)):
            try:
                shutil.rmtree(os.path.join(db_model.trained_model_path, f))
            except Exception as e:
                logger.exception('Could not remove visualization cache. {0}'.format(e))
                raise e

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

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
            batchsize,
            val_batchsize,
            model,
            loaderjob,
            db_model.epoch,
            optimizer,
            data_q,
            db_model.avoid_flipping
        )
    )
    feeder.daemon = True
    feeder.start()
    train_logger = threading.Thread(
        target=log_result,
        args=(
            batchsize,
            val_batchsize,
            os.path.join(db_model.trained_model_path, 'line_graph.tsv'),
            os.path.join(db_model.trained_model_path, 'log.html'),
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
        data_q
    )
    feeder.join()
    train_logger.join()

    # post-processing
    _post_process(db_model, pretrained_model)
    logger.info('Finish imagenet train. model_id: {0}'.format(db_model.id))

def do_train_by_tensorflow(
    db_model,
    output_dir_root,
    batchsize,
    val_batchsize,
    pretrained_model,
    gpu_num
):
    logger.info('Start imagenet train. model_id: {}, pretrained_model: {}'.format(db_model.id, pretrained_model))

    train_list = load_image_list(os.path.join(db_model.prepared_file_path, 'train.txt'))
    val_list   = load_image_list(os.path.join(db_model.prepared_file_path, 'test.txt'))
    mean_image = pickle.load(open(os.path.join(db_model.prepared_file_path, 'mean.npy'), 'rb'))

    # load image and labels
    flip = True if db_model.avoid_flipping == 0 else False
    num_classes = 1000
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    for t in train_list:
        #temp_image = np.asarray(Image.open(t[0]))
        temp_image = read_image(t[0], 128, mean_image, center=False, flip=flip, original_size=128)
        train_images.append(temp_image.flatten().astype(np.float32) / 255.0)
        train_labels.append(t[1])
    for v in val_list:
        #temp_image = np.asarray(Image.open(v[0]))
        temp_image = read_image(v[0], 128, mean_image, center=True, flip=False, original_size=128)
        val_images.append(temp_image.flatten().astype(np.float32) / 255.0)
        val_labels.append(v[1])
    train_labels_one_hot = []
    val_labels_one_hot = []
    for l in train_labels:
        tmp = np.zeros(num_classes)
        tmp[int(l)] = 1
        train_labels_one_hot.append(tmp)
    for l in val_labels:
        tmp = np.zeros(num_classes)
        tmp[int(l)] = 1
        val_labels_one_hot.append(tmp)

    # load model
    (model_dir, model_name) = os.path.split(db_model.network_path)
    model_name = re.sub(r"\.py$", "", model_name)
    model = load_module(model_dir, model_name)

    db_model.trained_model_path = _create_trained_model_dir(db_model.trained_model_path, output_dir_root, model_name)

    db_model.is_trained = 1
    db_model.update_and_commit()

    if gpu_num > -1:
        device = '/gpu:' + str(gpu_num)
    else:
        device = '/cpu:0'

    with tf.device(device):
        images_placeholder = tf.placeholder(tf.float32, [None, 128 * 128 * 3])
        labels_placeholder = tf.placeholder(tf.float32, [None, num_classes])
        keep_prob = tf.placeholder(tf.float32)
        trainable = tf.placeholder(tf.bool)

        logits = model.inference(images_placeholder, keep_prob, tf.bool)
        loss_value = model.loss(logits, labels_placeholder)
        train_op = model.training(loss_value, 1e-4)
        acc = model.accuracy(logits, labels_placeholder)

    saver = tf.train.Saver()

    with open(os.path.join(db_model.trained_model_path, 'line_graph.tsv'), 'w') as line_graph, open(os.path.join(db_model.trained_model_path, 'log.html'), 'w') as log_file:
        line_graph.write("count\tepoch\taccuracy\tloss\taccuracy(val)\tloss(val)\n")
        line_graph.flush()
        counter = 0
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for step in range(db_model.epoch):
                for i in range(len(train_images)/batchsize):
                    batch = batchsize * i
                    sess.run(train_op, feed_dict={
                        images_placeholder: train_images[batch:batch+batchsize],
                        labels_placeholder: train_labels_one_hot[batch:batch+batchsize],
                        keep_prob: 0.5,
                        trainable: True
                    })
                    counter += 1

                train_accuracy, train_loss = sess.run([acc, loss_value], feed_dict={
                    images_placeholder: train_images,
                    labels_placeholder: train_labels_one_hot,
                    keep_prob: 1.0,
                    trainable: True
                })
                line_graph.write('{}\t{}\t{}\t{}\t\t\n'.format(counter, step, train_accuracy, train_loss))
                line_graph.flush()
                log_file.write("[TIME]{},{}<br>".format(step+1,datetime.datetime.now().strftime( '%Y-%m-%d %H:%M:%S' )))
                log_file.write("[TRAIN] epoch {}, loss {}, acc {}<br>".format(step, train_loss, train_accuracy))
                log_file.flush()
                val_accuracy, val_loss = sess.run([acc, loss_value], feed_dict={
                    images_placeholder: val_images,
                    labels_placeholder: val_labels_one_hot,
                    keep_prob: 1.0,
                    trainable: False
                })
                line_graph.write('{}\t{}\t\t\t{}\t{}\n'.format(counter, step, val_accuracy, val_loss))
                line_graph.flush()
                log_file.write("[VALIDATION] ecpoch {}, loss {}, acc {}<br>".format(step, val_loss, val_accuracy))
                log_file.flush()

                save_path = saver.save(sess, os.path.join(db_model.trained_model_path, 'model{:0>4}.ckpt'.format(step)))

    # post-processing
    _post_process(db_model, pretrained_model)
    logger.info('Finish imagenet train. model_id: {0}'.format(db_model.id))
