#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import math
import sys
import argparse
import cPickle as pickle

import os
import re
import imp

import random
import bisect

import numpy as np
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import codecs

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol, [dir_name])
    return imp.load_module(symbol, file, path, description)

def predict(model_path, vocab_path, network_path, primetext,seed, unit,dropout,sample,length, use_mecab=0,gpu=0):

    np.random.seed(seed)

    # load vocabulary
    vocab = pickle.load(open(vocab_path, 'rb'))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c

    n_units = unit

    network = network_path.split(os.sep)[-1]
    model_name = re.sub(r"\.py$", "", network)
    model_module = load_module(os.path.dirname(args.network), model_name)
    lm = model_module.Network(len(vocab), n_units, dropout_ratio=dropout, train=False)
 
    model = L.Classifier(lm)
    model.compute_accuracy = False  # we only want the perplexity
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    serializers.load_npz(model_path, model)

    if args.gpu >= 0:
        cuda.init()
        model.to_gpu()

    model.predictor.reset_state()  # initialize state

    global prev_char

    prev_char = np.array([0])
    if args.gpu >= 0:
        prev_char = cuda.to_gpu(prev_char)

    ret = []

    if use_mecab == 1:
        if len(primetext) > 0:
            for i in unicode(primetext, 'utf-8'):
                ret.append(i)
                prev_char = Variable(np.ones((1,)).astype(np.int32) * vocab[i])
                if args.gpu >= 0:
                    prev_char = cuda.to_gpu(prev_char)

                prob = model.predictor.predict(prev_char)
    else:
        if len(primetext) > 0:
            prev_char = Variable(np.ones((1,)).astype(np.int32) * vocab[primetext])
            if args.gpu >= 0:
                prev_char = cuda.to_gpu(prev_char)
                    
            prob = F.softmax(model.predictor(prev_char))
            ret.append(primetext)

    for i in xrange(args.length):
        prob = model.predictor.predict(prev_char)

        if sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))

        if ivocab[index] == "<eos>":
            et.append(".")
        else:
            et.append(ivocab[index])

        prev_char = Variable(np.ones((1,)).astype(np.int32) * vocab[ivocab[index]])
        if args.gpu >= 0:
            prev_char = cuda.to_gpu(prev_char)

    return "".join(ret)

