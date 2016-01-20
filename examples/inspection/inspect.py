import os
import argparse
import numpy as np
import json
import random
import re
import imp

from PIL import Image

import six
import cPickle as pickle

import chainer
from chainer import cuda
from chainer import serializers

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol, [dir_name])
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

def inspect(image_path, mean, model_path, label, network, gpu=-1):
    network = network.split(os.sep)[-1]
    model_name = re.sub(r"\.py$", "", network)
    model_module = load_module(os.path.dirname(model_path), model_name)
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do inspection by command line')
    parser.add_argument('image_to_inspect', help='Path to the image file which you want to inspect')
    parser.add_argument('network', help='Path to the network model file')
    parser.add_argument('model', help='Path to the trained model (downloaded from DEEPstation ')
    parser.add_argument('--label', '-l', default='labels.txt',
                         help='Path to the labels.txt file (downloaded from DEEPstation)')
    parser.add_argument('--mean', '-m', default='mean.npy',
                         help='Path to the mean file (downloaded from DEEPstation)')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    results = inspect(args.image_to_inspect, args.mean, args.model, args.label, args.network, args.gpu)
    print "{rank:<5}:{name:<40} {score}".format(rank='Rank', name='Name', score='Score')
    print "----------------------------------------------------"
    for result in results:
        print "{rank:<5}:{name:<40} {score}".format(**result)
