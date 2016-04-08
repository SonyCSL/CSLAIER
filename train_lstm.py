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
import shutil

import numpy as np
import six
import cPickle as pickle
import copy
import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer import link

# gotten from http://qiita.com/tabe2314/items/6c0c1b769e12ab1e2614
def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print ('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print ('Copy %s' % child.name)

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol,[dir_name])
    return imp.load_module(symbol, file, path, description)

def load_data(filename, use_mecab,vocab):
    if use_mecab:
        words = codecs.open(filename, 'rb', 'utf-8').read().replace('\n', '<eos>').strip().split()
    else:
        words = codecs.open(filename, 'rb', 'utf-8').read()
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
    vocabulary,
    use_mecab = False,
    initmodel = None,
    resume = None,
    gpu = -1,
    rnn_size = 128,
    learning_rate = 2e-3,
    learning_rate_decay = 0.97,
    learning_rate_decay_after = 10,
    decay_rate = 0.95,
    dropout = 0.0,
    seq_length = 50,
    batchsize = 50,
    epochs = 50,
    grad_clip = 5
):
    n_epoch = epochs if isinstance(epochs, int) else int(epochs, 10)  # number of epochs
    n_units = rnn_size  # number of units per layer
    batchsize = batchsize   # minibatch size
    bprop_len = seq_length   # length of truncated BPTT
    grad_clip = grad_clip    # gradient norm threshold to clip
    
    conn = sqlite3.connect(db_path)
    db = conn.cursor()
    cursor = db.execute('select name from Model where id = ?', (model_id,))
    row = cursor.fetchone()
    model_name = row[0]
    
    model_name = re.sub(r"\.py$","", model_name)
    model_module = load_module(model_dir, model_name)

    output_dir = root_output_dir + os.sep + model_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    vocab = {}
    vocab_size = 0
    
    if vocabulary != '':
        vocab = pickle.load(open(vocabulary, 'rb'))
        vocab_size = len(vocab)
    
    if gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu >= 0 else np
    
    train_data, words, vocab = load_data(filename, use_mecab,vocab)
    pickle.dump(vocab, open('%s/vocab2.bin'%output_dir, 'wb'))
    
    # Prepare model
    lm = model_module.Network(len(vocab), rnn_size, dropout_ratio=dropout, train=False)
    model = L.Classifier(lm)
    model.compute_accuracy = False  # we only want the perplexity
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    # Setup optimizer
    optimizer = optimizers.RMSprop(lr=learning_rate, alpha=decay_rate, eps=1e-8)
    optimizer.setup(model)
    
    # Load pretrained model
    if initmodel is not None and initmodel.find("model") > -1:
        if vocabulary == '':
            print("Load model from : "+output_dir + os.sep + initmodel)
            serializers.load_npz(output_dir + os.sep + initmodel, model)
        else:
            lm2 = model_module.Network(vocab_size, rnn_size, dropout_ratio=dropout, train=False)
            model2 = L.Classifier(lm2)
            model2.compute_accuracy = False  # we only want the perplexity
            print("Load model from : "+output_dir + os.sep + initmodel)
            serializers.load_npz(output_dir + os.sep + initmodel, model2)
            copy_model(model2,model)
        # delete old models
        shutil.copyfile(output_dir + os.sep + initmodel, output_dir + os.sep + 'previous_' + initmodel)
        pretrained_models = sorted(os.listdir(output_dir), reverse=True)
        for m in pretrained_models:
            if m.startswith('model') and initmodel != m:
                os.remove(output_dir + os.sep + m)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
        
    # Load pretrained optimizer
    if resume is not None and resume.find("state") > -1:
        print("Load optimizer state from : "+output_dir + os.sep + resume)
        serializers.load_npz(output_dir + os.sep +resume, optimizer)
    # TODO: delete old states ??
    
    # delete layer visualization cache
    for f in os.listdir(output_dir):
        if os.path.isdir(output_dir + os.sep + f):
            shutil.rmtree(output_dir + os.sep + f)
    
    use_wakatigaki = 1 if use_mecab else 0
    db.execute('update Model set epoch = ?, trained_model_path = ?, is_trained = 1, line_graph_data_path = ?, use_wakatigaki = ? where id = ?', (n_epoch, output_dir, output_dir + os.sep + 'line_graph.tsv', use_wakatigaki, model_id))
    conn.commit()
    
    log_file = open(output_dir + os.sep + 'log.html', 'w')
    #graph_tsv = open(output_dir + os.sep + 'line_graph.tsv', 'w')
    
    # Learning loop
    whole_len = train_data.shape[0]
    jump = whole_len // batchsize
    cur_log_perp = xp.zeros(())
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    accum_loss = 0
    batch_idxs = list(range(batchsize))
    log_file.write("going to train {} iterations<br>".format(jump * n_epoch))
    log_file.flush()
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
            log_file.write('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)<br>'.format(i + 1, perp, throuput))
            log_file.flush()
            cur_at = now
            cur_log_perp.fill(0)
    
        if (i + 1) % jump == 0:
            epoch += 1
            now = time.time()
            cur_at += time.time() - now  # skip time of evaluation
        
            if epoch >= 6:
                optimizer.lr /= 1.2
                log_file.write('learning rate = {:.10f}<br>'.format(optimizer.lr))
                log_file.flush()
            # Save the model and the optimizer
            serializers.save_npz(output_dir + os.sep + 'model%04d'%epoch, model)
            log_file.write('--- epoch: {} ------------------------<br>'.format(epoch))
            log_file.flush()
            serializers.save_npz(output_dir + os.sep + 'rnnlm.state', optimizer)

        sys.stdout.flush()
    os.remove(output_dir + os.sep + 'previous_' + initmodel) # delete backup file
    log_file.write('===== finish train. =====')
    log_file.close()
    #graph_tsv.close()
    db.execute('update Model set is_trained = 2, pid = null where id = ?', (model_id,))
    conn.commit()
    db.close()
    
    return