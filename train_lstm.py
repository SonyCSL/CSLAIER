# -*- coding: utf-8 -*-

from __future__ import print_function
import math
import sys
import time
import os
import codecs
import imp
import re
import sqlite3

import numpy as np
import six
import cPickle as pickle
import copy
import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol,[dir_name])
    return imp.load_module(symbol, file, path, description)

def load_data(filename, use_mecab):
    vocab = {}
    if use_mecab == 0:
        words = open(args.filename).read().replace('\n', '<eos>').strip().split()
    else:
        words = codecs.open(args.filename, 'rb', 'utf-8').read()
        words = list(words)
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    print('corpus length:', len(words))
    print('vocab size:', len(vocab))
    return dataset, words, vocab

def train_lstm(
    db_path,
    model_id,
    model_dir,
    root_output_dir,
    filename,
    data_dir,
    use_mecab = 0,
    initmodel = None,
    resume = None,
    gpu = -1,
    rnn_size = 128,
    learning_rate = 2e-3,
    learning_rate_decay = 0.97,
    learning_rate_decay_after = 10
    decay_rate = 0.95,
    dropout = 0.0,
    seq_length = 50,
    batchsize = 50,
    epoch = 50,
    grad_clip = 5
):
    conn = sqlite3.connect(db_path)
    db = conn.cursor()
    cursor = db.execute('select name from Model where id = ?', (model_id,))
    row = cursor.fetchone()
    model_name = row[0]
    
    model_name = re.sub(r"\.py$", model_name)
    model_module = load_module(model_dir, model_name)
    
    output_dir = root_output_dir + os.sep + model_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu >= 0 else np
    
    train_data, words, vocab = load_data(filename)
    pickle.dump(vocab, open('%s/vocab2.bin'%data_dir, 'wb'))
    
    # Prepare model
    lm = model_module.Network(len(vocab), rnn_size, dropout_ratio=args.dropout, train=False)
    model = L.Classifier(lm)
    model.compute_accuracy = False  # we only want the perplexity
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.RMSprop(lr=learning_rate, alpha=decay_rate, eps=1e-8)
    optimizer.setup(model)
    
    # Load pretrained model
    if initmodel is not None and initmodel.find("model") > -1:
        print("Load model from : "+output_dir + os.sep + initmodel)
        serializers.load_npz(output_dir + os.sep + initmodel, model)
        # TODO: delete old models ??
        
    # Load pretrained model
    if resume is not None and resume.find("state") > -1:
        print("Load optimizer state from : "+output_dir + os.sep + resume)
        serializers.load_npz(output_dir + os.sep +pretrained_model, model)
    # TODO: delete old states ??

    # Learning loop
    whole_len = train_data.shape[0]
    jump = whole_len // batchsize
    cur_log_perp = xp.zeros(())
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    accum_loss = 0
    batch_idxs = list(range(batchsize))
    print('going to train {} iterations'.format(jump * n_epoch))

    for i in six.moves.range(jump * n_epoch):
        x = chainer.Variable(xp.asarray(
                                    [train_data[(jump * j + i) % whole_len] for j in batch_idxs]))
        t = chainer.Variable(xp.asarray(
                                    [train_data[(jump * j + i + 1) % whole_len] for j in batch_idxs]))
        loss_i = model(x, t)
        accum_loss += loss_i
        cur_log_perp += loss_i.data
    
        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            accum_loss = 0
            optimizer.update()
    
        if (i + 1) % 10000 == 0:
            now = time.time()
            throuput = 10000. / (now - cur_at)
            perp = math.exp(float(cur_log_perp) / 10000)
            print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
                                                                              i + 1, perp, throuput))
            cur_at = now
            cur_log_perp.fill(0)
    
        if (i + 1) % jump == 0:
            epoch += 1
            now = time.time()
            cur_at += time.time() - now  # skip time of evaluation
        
            if epoch >= 6:
                optimizer.lr /= 1.2
                print('learning rate =', optimizer.lr)
    
        sys.stdout.flush()


    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('rnnlm.model', model)
    print('save the optimizer')
    serializers.save_npz('rnnlm.state', optimizer)
    
    return