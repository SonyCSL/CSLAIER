import os
import numpy as np
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time
import re
import imp

import numpy as np
from PIL import Image

import six
import cPickle as pickle
from six.moves import queue

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import h5py

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(dir_name + os.sep + symbol)
    return imp.load_module(symbol, file, path, description)

def read_image(path, model, mean_image, cropwidth, center=False, flip=False):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    if center:
        top = left = cropwidth / 2
    else:
        top = random.randint(0, cropwidth - 1)
        left = random.randint(0, cropwidth - 1)
    bottom = model.insize + top
    right = model.insize + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    image -= mean_image[:, top:bottom, left:right]
    image /= 255
    if flip and random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image

def inspect(image_path, mean, model_path, label, network, gpu=0):
    network = network.split(os.sep)[-1]
    model_name = re.sub(r"\.py$", "", network)
    model_module = load_module('models', model_name)
    mean_image = pickle.load(open(mean, 'rb'))
    model = model_module.Network()
    serializers.load_hdf5(model_path, model)
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        model.to_gpu()
    cropwidth = 256 - model.insize
    img = read_image(image_path, model, mean_image, cropwidth)
    x = np.ndarray((1, 3, model.insize, model.insize), dtype=np.float32)
    x[0] = img
    if gpu >= 0:
        x = cuda.to_gpu(x)
    score = model.predict(x)
    score = cuda.to_cpu(score.data)
    categories = np.loadtxt(label, str, delimiter="\t")
    top_k = 20
    prediction = zip(score[0].tolist(), categories)
    prediction.sort(cmp=lambda x, y:cmp(x[0], y[0]), reverse=True)
    ret = []
    for rank, (score, name) in enumerate(prediction[:top_k], start=1):
        ret.append({"rank": rank, "name": name, "score": "{0:4.1f}%".format(score*100)})
    return ret
    
