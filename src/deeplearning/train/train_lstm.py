# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import math
import sys
import time
import os
import codecs
import imp
import re
import shutil
import datetime
from logging import getLogger

import numpy as np
import six
import cPickle as pickle
import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer import link

# from .utils import remove_resume_file


logger = getLogger(__name__)


# gotten from http://qiita.com/tabe2314/items/6c0c1b769e12ab1e2614
def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
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
                logger.info('Ignore {0} because of parameter mismatch'.format(child.name))
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            logger.info('Copy {0}'.format(child.name))


def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol, [dir_name])
    return imp.load_module(symbol, file, path, description)


def load_data(filename, use_mecab, vocab):
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
    logger.info('corpus length: {0}'.format(len(words)))
    logger.info('vocab size: {0}'.format(len(vocab)))
    return dataset, words, vocab


def do_train(
        db_model,
        root_output_dir,
        filename,
        vocabulary,
        use_mecab=False,
        initmodel=None,
        resume=False,
        rnn_size=128,
        learning_rate=2e-3,
        learning_rate_decay=0.97,
        learning_rate_decay_after=10,
        decay_rate=0.95,
        dropout=0.0,
        seq_length=50,
        batchsize=50,  # minibatch size
        grad_clip=5,  # gradient norm threshold to clip
        interruptable=None
):
    logger.info('Start LSTM training. model_id: {0}, use_mecab: {1}, initmodel: {2}, gpu: {3}'
                .format(db_model.id, use_mecab, initmodel, db_model.gpu))
    n_epoch = db_model.epoch
    bprop_len = seq_length  # length of truncated BPTT
    grad_clip = grad_clip

    (model_dir, model_name) = os.path.split(db_model.network_path)
    model_name = re.sub(r"\.py$", "", model_name)
    model_module = load_module(model_dir, model_name)

    if db_model.trained_model_path is None:
        db_model.trained_model_path = os.path.join(root_output_dir, model_name)
    if not os.path.exists(db_model.trained_model_path):
        os.mkdir(db_model.trained_model_path)

    vocab = {}
    vocab_size = 0

    if vocabulary != '':
        vocab = pickle.load(open(vocabulary, 'rb'))
        vocab_size = len(vocab)

    if db_model.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if db_model.gpu >= 0 else np

    train_data, words, vocab = load_data(filename, use_mecab, vocab)
    pickle.dump(vocab, open('%s/vocab2.bin' % db_model.trained_model_path, 'wb'))

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
            logger.info("Load model from : " + db_model.trained_model_path + os.sep + initmodel)
            serializers.load_npz(os.path.join(db_model.trained_model_path, initmodel), model)
        else:
            lm2 = model_module.Network(vocab_size, rnn_size, dropout_ratio=dropout, train=False)
            model2 = L.Classifier(lm2)
            model2.compute_accuracy = False  # we only want the perplexity
            logger.info("Load model from : " + db_model.trained_model_path + os.sep + initmodel)
            serializers.load_npz(os.path.join(db_model.trained_model_path, initmodel), model2)
            copy_model(model2, model)
        # delete old models
        try:
            shutil.copyfile(os.path.join(db_model.trained_model_path, initmodel),
                            os.path.join(db_model.trained_model_path, 'previous_' + initmodel))
        except Exception as e:
            logger.exception('Could not copy {0} to {1}. {2}'
                             .format(os.path.join(db_model.trained_model_path, initmodel),
                                     os.path.join(db_model.trained_model_path,
                                                  'previous_' + initmodel), e))
            raise e
        pretrained_models = sorted(os.listdir(db_model.trained_model_path), reverse=True)
        for m in pretrained_models:
            if m.startswith('model') and initmodel != m:
                try:
                    os.remove(os.path.join(db_model.trained_model_path, m))
                except Exception as e:
                    logger.exception('Could not remove old trained model: {0} {1}'
                                     .format(os.path.join(db_model.trained_model_path, m), e))
                    raise e

    if db_model.gpu >= 0:
        cuda.get_device(db_model.gpu).use()
        model.to_gpu()

    # Load pretrained optimizer
    resume_path = os.path.join(db_model.trained_model_path, 'resume')
    if resume:
        logger.info("Load optimizer state from : {}".format(os.path.join(db_model.trained_model_path, 'resume.state')))
        serializers.load_npz(os.path.join(resume_path, 'resume.model'), model)
        serializers.load_npz(os.path.join(resume_path, 'resume.state'), optimizer)

    db_model.is_trained = 1
    db_model.update_and_commit()

    # Learning loop
    whole_len = train_data.shape[0]
    jump = whole_len // batchsize
    if resume:
        resume_data = json.load(open(os.path.join(resume_path, 'resume.json')))
        initmodel = resume_data['initmodel']
        cur_log_perp = xp.zeros(())
        cur_log_perp += resume_data['cur_log_perp']
        loss_for_graph = xp.zeros(())
        loss_for_graph += resume_data['loss_for_graph']
        iteration_from = resume_data['i']
        epoch = (iteration_from + 1) / jump
    else:
        cur_log_perp = xp.zeros(())
        loss_for_graph = xp.zeros(())
        iteration_from = 0
        epoch = 0

    start_at = time.time()
    cur_at = start_at
    accum_loss = 0
    batch_idxs = list(range(batchsize))

    graph_tsv_path = os.path.join(db_model.trained_model_path, 'line_graph.tsv')
    train_log_path = os.path.join(db_model.trained_model_path, 'train.log')
    if not resume:
        with open(graph_tsv_path, 'a') as fp:
            fp.write('count\tepoch\tperplexity\n')

        with open(train_log_path, 'a') as fp:
            fp.write(json.dumps({
                'type': 'text',
                'text': "going to train {} iterations".format(jump * n_epoch)
            }) + '\n')

    # delete layer visualization cache
    # trained_model_pathに存在する全てのディレクトリを削除している。
    for f in os.listdir(db_model.trained_model_path):
        if os.path.isdir(os.path.join(db_model.trained_model_path, f)):
            try:
                shutil.rmtree(os.path.join(db_model.trained_model_path, f))
            except Exception as e:
                logger.exception('Could not remove visualization cache: {0} {1}'
                                 .format(os.path.join(db_model.trained_model_path, f), e))
                raise e
    # ので、↓のresumeファイルの削除は不要
    # remove_resume_file(db_model.trained_model_path)

    for i in six.moves.range(iteration_from, jump * n_epoch):
        # 1バッチが終わったタイミングを意図している。
        if interruptable.is_interrupting() and isinstance(accum_loss, int):
            os.mkdir(resume_path)
            serializers.save_npz(os.path.join(resume_path, 'resume.state'), optimizer)
            serializers.save_npz(os.path.join(resume_path, 'resume.model'), model)
            json.dump({
                'i': i,
                'initmodel': initmodel,
                'cur_log_perp': float(cur_log_perp),
                'loss_for_graph': float(loss_for_graph),
                'epoch': epoch
            }, open(os.path.join(resume_path, 'resume.json'), 'w'))
            interruptable.set_interruptable()
            while True:
                time.sleep(1)
        x = chainer.Variable(
            xp.asarray([train_data[(jump * j + i) % whole_len] for j in batch_idxs]))
        t = chainer.Variable(
            xp.asarray([train_data[(jump * j + i + 1) % whole_len] for j in batch_idxs]))
        loss_i = model(x, t)
        accum_loss += loss_i
        loss_for_graph += loss_i.data
        cur_log_perp += loss_i.data

        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            accum_loss = 0
            optimizer.update()

        if (i + 1) % 100 == 0:
            now = time.time()
            throuput = 10000. / (now - cur_at)
            perp = math.exp(float(cur_log_perp) / 10000)
            with open(train_log_path, 'a') as fp:
                fp.write(json.dumps({
                    'type': 'log',
                    'log': 'iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(i + 1, perp, throuput),
                    'time_stamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'epoch': epoch
                }) + '\n')
            cur_at = now
            cur_log_perp.fill(0)

        if (i + 1) % 100 == 0:
            perp_for_graph = math.exp(float(loss_for_graph) / 100)
            with open(graph_tsv_path, 'a') as fp:
                fp.write('{}\t{}\t{:.2f}\n'.format(i + 1, epoch, perp_for_graph))
            loss_for_graph.fill(0)

        if (i + 1) % jump == 0:
            epoch += 1
            now = time.time()
            cur_at += time.time() - now  # skip time of evaluation

            with open(train_log_path, 'a') as fp:
                if epoch >= 6:
                    optimizer.lr /= 1.2
                    fp.write(json.dumps({
                        'type': 'data',
                        'text': 'learning rate = {:.10f}'.format(optimizer.lr),
                    }) + '\n')
                fp.write(json.dumps({
                    'type': 'text',
                    'text': '--- epoch: {} ------------------------'.format(epoch),
                }) + '\n')
            # Save the model and the optimizer
            serializers.save_npz(os.path.join(db_model.trained_model_path,
                                              'model%04d' % epoch), model)
            serializers.save_npz(os.path.join(db_model.trained_model_path,
                                              'rnnlm.state'), optimizer)

        sys.stdout.flush()
    if os.path.exists(os.path.join(db_model.trained_model_path, 'previous_' + initmodel)):
        # delete backup file
        try:
            os.remove(os.path.join(db_model.trained_model_path, 'previous_' + initmodel))
        except Exception as e:
            logger.exception('Could not remove backuped file: {0} {1}'
                             .format(os.path.join(db_model.trained_model_path,
                                                  'previous_' + initmodel), e))
            raise e
    with open(train_log_path, 'a') as fp:
        fp.write(json.dumps({
            'type': 'text',
            'text': '===== finish train. =====',
        }) + '\n')
    db_model.is_trained = 2
    db_model.pid = None
    db_model.gpu = None
    db_model.update_and_commit()
    interruptable.clear_interrupt()
    interruptable.terminate()
    logger.info('Finish LSTM train. model_id: {0}'.format(db_model.id))
