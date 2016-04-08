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
import sqlite3
import shutil

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

VALIDATION_TIMING = 500 # ORIGINAL 50000

def load_image_list(path):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((pair[0], np.int32(pair[1])))
    return tuples

def read_image(path, model_insize, mean_image, center=False, flip=False):
    cropwidth = 256 - model_insize
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    if center:
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
        print('epoch', epoch, file=sys.stderr)
        print('learning rate', optimizer.lr, file=sys.stderr)
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
            if count % 1000 == 0:
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
                '\rtrain {} updates ({} samples) time: {} ({} images/sec)<BR/>'
                .format(train_count, train_count * batchsize, datetime.timedelta(seconds=duration), throughput))
            fH.flush()
            f.write(str(count) + "\t" + str(epoch) + "\t" + str(accuracy) + "\t" + str(loss) + "\t\t\n")
            f.flush()
            count += 1
            train_cur_loss += loss
            train_cur_accuracy += accuracy
            if train_count % 1000 == 0:
                mean_loss = train_cur_loss / 1000
                mean_error = 1 - train_cur_accuracy / 10000
                fH.write("<strong>"+json.dumps({'type': 'train', 'iteration': train_count, 'error': mean_error, 'loss': mean_loss})+"</strong><br/>")
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
                fH.write("<strong>"+json.dumps({'type': 'val', 'iteration': train_count, 'error': (1 - mean_accuracy), 'loss': mean_loss})+"</strong><br/>")
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
                print('generated graph')
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
            
def do_train(db_path, train, test, mean, root_output_dir, model_dir, model_id, batchsize=32, val_batchsize=250, epoch=10, gpu=-1, loaderjob=20,pretrained_model="",avoid_flipping=0):
    conn = sqlite3.connect(db_path)
    db = conn.cursor()
    cursor = db.execute('select name from Model where id = ?', (model_id,))
    row = cursor.fetchone()
    model_name = row[0]
    # start initialization
    if gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu >= 0 else np
    
    train_list = load_image_list(train)
    val_list = load_image_list(test)
    mean_image = pickle.load(open(mean, 'rb'))
    
    # @see http://qiita.com/progrommer/items/abd2276f314792c359da
    model_name = re.sub(r"\.py$", "", model_name)
    model_module = load_module(model_dir, model_name)
    model = model_module.Network()
    
    # create directory for saving trained models
    output_dir = root_output_dir + os.sep + model_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Load pretrained model
    if pretrained_model is not None and pretrained_model.find("model") > -1:
        print("load pretrained model : "+output_dir + os.sep +pretrained_model)
        serializers.load_hdf5(output_dir + os.sep +pretrained_model, model)
        # delete old models
        shutil.copyfile(output_dir + os.sep + pretrained_model, output_dir + os.sep + 'previous_' + pretrained_model)
        pretrained_models = sorted(os.listdir(output_dir), reverse=True)
        for m in pretrained_models:
            if m.startswith('model') and pretrained_model != m:
                os.remove(output_dir + os.sep + m)
                
    # delete layer visualization cache
    for f in os.listdir(output_dir):
        if os.path.isdir(output_dir + os.sep + f):
            shutil.rmtree(output_dir + os.sep + f)
    
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
        
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    
    data_q = queue.Queue(maxsize=1)
    res_q = queue.Queue()
    db.execute('update Model set epoch = ?, trained_model_path = ?, graph_data_path = ?, is_trained = 1, line_graph_data_path = ? where id = ?', (epoch, output_dir, output_dir + os.sep + 'graph.dot', output_dir + os.sep + 'line_graph.tsv', model_id))
    conn.commit()
    
    # Invoke threads
    feeder = threading.Thread(target=feed_data, args=(train_list, val_list, mean_image, batchsize, val_batchsize, model, loaderjob, epoch, optimizer, data_q, avoid_flipping))
    feeder.daemon = True
    feeder.start()
    logger = threading.Thread(target=log_result, args=(batchsize, val_batchsize, output_dir + os.sep + 'line_graph.tsv',output_dir + os.sep + 'log.html',  res_q))
    logger.daemon = True
    logger.start()
    train_loop(model, output_dir, xp, optimizer, res_q, data_q)
    feeder.join()
    logger.join()
    db.execute('update Model set is_trained = 2, pid = null where id = ?', (model_id,))
    conn.commit()
    db.close()
    os.remove(output_dir + os.sep + 'previous_' + pretrained_model) #delete backup file
