import os
import numpy as np
import json
import random
import re
import imp

from PIL import Image
import scipy.misc
import cv2

import six
import cPickle as pickle

import chainer
from chainer import cuda
from chainer import serializers

def load_module(dir_name, symbol):
    (file, path, description) = imp.find_module(symbol, [dir_name])
    return imp.load_module(symbol, file, path, description)

def read_image(path, height, width, resize_mode = "squash", channels=3, flip=False):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    channels -- the PIL mode that the image should be converted to
        (3 for color or 1 for grayscale)
    resize_mode -- can be crop, squash, fill or half_crop
    flip -- flag for flipping
    """
    if channels == 1:
        mode = "L"
    else:
        mode = "RGB"
        
    image = Image.open(path)
    image = image.convert(mode)
    image = np.array(image)

    ### Resize
    interp = 'bilinear'
    
    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if resize_mode == 'squash' or width_ratio == height_ratio:
        return scipy.misc.imresize(image, (height, width), interp=interp)
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)

        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width-width)/2.0))
            return image[:,start:start+width]
        else:
            start = int(round((resize_height-height)/2.0))
            return image[start:start+height,:]
    else:
        if resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = width
                resize_height = int(round(image.shape[0] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.shape[1] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.shape[1] / new_ratio))
            resize_height = int(round(image.shape[0] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width-width)/2.0))
                image = image[:,start:start+width]
            else:
                start = int(round((resize_height-height)/2.0))
                image = image[start:start+height,:]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = int((height - resize_height)/2)
            noise_size = (padding, width)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=0)
        else:
            padding = int((width - resize_width)/2)
            noise_size = (height, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=1)

    if flip and random.randint(0, 1) == 0:
        return np.fliplr(image)
    else:
        return image


def inspect(image_path, mean, model_path, label, network_path, resize_mode,channels, gpu=0):
    network = network_path.split(os.sep)[-1]
    model_name = re.sub(r"\.py$", "", network)
    model_module = load_module(os.path.dirname(network_path), model_name)
    mean_image = pickle.load(open(mean, 'rb'))
    model = model_module.Network()
    serializers.load_hdf5(model_path, model)
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        model.to_gpu()
    
    output_side_length = 256
        
    img = read_image(image_path, 256, 256, resize_mode,channels)

    cropwidth = 256 - model.insize
    top = left = cropwidth / 2
    bottom = model.insize + top
    right = model.insize + left
    if img.ndim == 3:
        img = img.transpose(2, 0, 1)
        img = img[:, top:bottom, left:right].astype(np.float32)
    else:
        img = img[top:bottom, left:right].astype(np.float32)
        zeros = np.zeros((model.insize,model.insize))
        img = np.array([img, zeros, zeros])
    img -= mean_image[:, top:bottom, left:right]
    img /= 255
    
    x = np.ndarray((1, 3,  model.insize, model.insize), dtype=np.float32)
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
    

