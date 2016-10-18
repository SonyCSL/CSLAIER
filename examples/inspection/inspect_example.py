import os
import sys
import argparse
import numpy as np
import random
import re
import imp
from PIL import Image
import scipy.misc
import cPickle as pickle
from chainer import cuda
from chainer import serializers
try:
    import tensorflow as tf
except ImportError:
    pass


def load_module(target):
    network = target.split(os.sep)[-1]
    model_name = re.sub(r"\.py$", "", network)
    (file, path, description) = imp.find_module(model_name, [os.path.dirname(target)])
    return imp.load_module(model_name, file, path, description)


def read_image(path, height, width,
               resize_mode="squash", channels=3, flip=False):
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

    # Resize
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
            return image[:, start:start+width]
        else:
            start = int(round((resize_height-height)/2.0))
            return image[start:start+height, :]
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
                image = image[:, start:start+width]
            else:
                start = int(round((resize_height-height)/2.0))
                image = image[start:start+height, :]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = int((height - resize_height)/2)
            noise_size = (padding, width)
            if channels > 1:
                noise_size += (channels,)
            print noise_size
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=0)
        else:
            padding = (width - resize_width)/2
            noise_size = (height, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=1)

    if flip and random.randint(0, 1) == 0:
        return np.fliplr(image)
    else:
        return image


def inspect_by_chainer(image_path, mean, model_path, label,
                       network_path, resize_mode, channels, gpu=-1):
    model_module = load_module(network_path)
    mean_image = pickle.load(open(mean, 'rb'))
    model = model_module.Network()
    serializers.load_hdf5(model_path, model)
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        model.to_gpu()

    img = read_image(image_path, 256, 256, resize_mode, channels)
    cropwidth = 256 - model.insize
    top = left = cropwidth / 2
    bottom = model.insize + top
    right = model.insize + left

    if img.ndim == 3:
        img = img.transpose(2, 0, 1)
        img = img[:, top:bottom, left:right].astype(np.float32)
    else:
        img = img[top:bottom, left:right].astype(np.float32)
        zeros = np.zeros((model.insize, model.insize))
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
    prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
    ret = []
    for rank, (score, name) in enumerate(prediction[:top_k], start=1):
        ret.append({"rank": rank, "name": name, "score": "{0:4.1f}%".format(score*100)})
    return ret


def inspect_by_tensorflow(image_path, mean, model_path, label,
                          network_path, resize_mode, channels, gpu=0):
    model_module = load_module(network_path)
    img = read_image(image_path, 128, 128, resize_mode, channels)
    if img.ndim == 3:
        img = img.transpose(2, 0, 1).astype(np.float32)
    else:
        zeros = np.zeros(128, 128)
        img = np.array([img, zeros, zeros]).astype(np.float32)
    mean_image = pickle.load(open(mean, 'rb'))
    img -= mean_image[:, 0:128, 0:128]
    feed_data = []
    feed_data.append(img.flatten().astype(np.float32) / 255.0)

    images_placeholder = tf.placeholder(tf.float32, [None, 128*128*3])
    keep_prob = tf.placeholder(tf.float32)

    logits = model_module.inference(images_placeholder, keep_prob)
    prediction_op = tf.nn.top_k(tf.nn.softmax(logits), k=20)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        (values, indeceis) = sess.run(prediction_op, feed_dict={
            images_placeholder: feed_data,
            keep_prob: 1.0,
        })

    categories = np.loadtxt(label, str, delimiter="\t")
    ret = []
    for i, idx in enumerate(indeceis[0]):
        ret.append({
            'rank': i+1,
            'name': categories[idx],
            'score': "{:4.1f}%".format(values[0][i]*100)
        })
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do inspection by command line')
    parser.add_argument('image_to_inspect', help='Path to the image file which you want to inspect')
    parser.add_argument('network', help='Path to the network model file')
    parser.add_argument('model', help='Path to the trained model (downloaded from CSLAIER ')
    parser.add_argument('--label', '-l', default='labels.txt',
                        help='Path to the labels.txt file (downloaded from CSLAIER)')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Path to the mean file (downloaded from CSLAIER)')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resize_mode', '-r', default='squash',
                        help='can be crop, squash, fill or half_crop')
    parser.add_argument('--channels', '-c', default='3',
                        help='3 for RGB or 1 for grayscale')
    parser.add_argument('--framework', '-f', default='chainer',
                        help='chainer or tensorflow')
    args = parser.parse_args()

    if args.framework == 'chainer':
        results = inspect_by_chainer(args.image_to_inspect, args.mean, args.model, args.label,
                                     args.network, args.resize_mode, int(args.channels), args.gpu)
    elif args.framework == 'tensorflow':
        results = inspect_by_tensorflow(args.image_to_inspect, args.mean, args.model, args.label,
                                        args.network, args.resize_mode, int(args.channels), args.gpu)
    else:
        print 'Unknown Framework'
        sys.exit()
    print "{rank:<5}:{name:<40} {score}".format(rank='Rank', name='Name', score='Score')
    print "----------------------------------------------------"
    for result in results:
        print "{rank:<5}:{name:<40} {score}".format(**result)
