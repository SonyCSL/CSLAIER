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

#%% arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m',type=str,   required=True,
                    help='model data, saved by train_ptb.py')
parser.add_argument('--vocabulary','-v',type=str,   required=True,
                    help='vocabulary data, saved by train_ptb.py')
parser.add_argument('--network', '-n', type=str,   required=True,
                    help='Path to the network model file')
parser.add_argument('--primetext', '-p', type=str,   default='',
                    help='base text data, used for text generation')
parser.add_argument('--seed', '-s', type=int,   default=123,
                    help='random seeds for text generation')
parser.add_argument('--unit', '-u',  type=int,   default=650,
                    help='number of units')
parser.add_argument('--dropout',    type=float, default=0.0,
                    help='dropout_ratio for the network')
parser.add_argument('--sample',     type=int,   default=1,
                    help='negative value indicates NOT use random choice')
parser.add_argument('--length',     type=int,   default=20,
                    help='length of the generated text')
parser.add_argument('--gpu',        type=int,   default=-1,
                    help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()

np.random.seed(args.seed)

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol, [dir_name])
    return imp.load_module(symbol, file, path, description)

# load vocabulary
vocab = pickle.load(open(args.vocabulary, 'rb'))
ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c

n_units = args.unit

network = args.network.split(os.sep)[-1]
model_name = re.sub(r"\.py$", "", network)
model_module = load_module(os.path.dirname(args.network), model_name)
lm = model_module.Network(len(vocab), n_units, dropout_ratio=args.dropout, train=False)
 
model = L.Classifier(lm)
model.compute_accuracy = False  # we only want the perplexity
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)

serializers.load_npz(args.model, model)

if args.gpu >= 0:
    cuda.init()
    model.to_gpu()

model.predictor.reset_state()  # initialize state

global prev_char

prev_char = np.array([0])
if args.gpu >= 0:
    prev_char = cuda.to_gpu(prev_char)

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

if len(args.primetext) > 0:
    for i in unicode(args.primetext, 'utf-8'):
        sys.stdout.write(i)
        prev_char = Variable(np.ones((1,)).astype(np.int32) * vocab[i])
        if args.gpu >= 0:
            prev_char = cuda.to_gpu(prev_char)

        prob = model.predictor.predict(prev_char)

for i in xrange(args.length):
    prob = model.predictor.predict(prev_char)

    if args.sample > 0:
        probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
        probability /= np.sum(probability)
        index = np.random.choice(range(len(probability)), p=probability)
    else:
        index = np.argmax(cuda.to_cpu(prob.data))

    sys.stdout.write(ivocab[index])

    prev_char = Variable(np.ones((1,)).astype(np.int32) * vocab[ivocab[index]])
    if args.gpu >= 0:
        prev_char = cuda.to_gpu(prev_char)

