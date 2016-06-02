# -*- encoding:utf-8 -*-
import os
from logging import getLogger

import numpy
import scipy
import scipy.ndimage
import cv2
import cPickle as pickle
from PIL import Image

import common.utils as ds_utils

logger = getLogger(__name__)

def do(model, prepared_data_root):
    if model.prepared_file_path:
        # re-use existing directory
        for f in os.listdir(model.prepared_file_path):
            os.remove(os.path.join(model.prepared_file_path, f))
    else:
        model.prepared_file_path = os.path.join(prepared_data_root, ds_utils.get_timestamp())
        os.mkdir(model.prepared_file_path)
    model.update_and_commit()
    logger.info('Start making training data.')
    make_train_data(model)
    compute_mean(model.prepared_file_path)
    logger.info('Finish making training data.')
    return model

def make_train_data(model):
    train_text  = open(os.path.join(model.prepared_file_path, 'train.txt'), 'w')
    test_text   = open(os.path.join(model.prepared_file_path, 'test.txt'), 'w')
    labels_text = open(os.path.join(model.prepared_file_path, 'labels.txt'), 'w')
    class_no = 0
    count = 0
    for path, dirs, files in os.walk(model.dataset.dataset_path):
        if not dirs:
            (head, tail) = os.path.split(path)
            labels_text.write(tail.encode('utf-8') + "\n")
            start_count = count
            length = len(files)
            for f in files:
                logger.info('Processing File: {0}'.format(os.path.join(path, f)))
                (head, ext) = os.path.splitext(f)
                ext = ext.lower()
                if ext not in [".jpg", ".jpeg", ".gif", ".png"]: continue
                if os.path.getsize(os.path.join(path, f)) <= 0: continue
                new_image_path = os.path.join(model.prepared_file_path, "image{0:0>7}.jpg".format(count))
                resize_image(os.path.join(path, f), new_image_path, model)
                if count - start_count < length * 0.75:
                    train_text.write("{0} {1:d}\n".format(new_image_path, class_no))
                else:
                    test_text.write("{0} {1:d}\n".format(new_image_path, class_no))
                count += 1
            class_no += 1
    train_text.close()
    test_text.close()
    labels_text.close()

def resize_image(source, dest, model):
    output_side_length = 256

    if model.channels == 1:
        mode = "L"
    else:
        mode = "RGB"

    image = Image.open(source)
    image = image.convert(mode)
    image = numpy.array(image)

    height = image.shape[0]
    width = image.shape[1]

    if model.resize_mode not in ['crop', 'squash', 'fill', 'half_crop']:
        raise ValueError('resize_mode "%s" not supported' % model.resize_mode)

	### Resize
    interp = 'bilinear'

    width_ratio = float(width) / output_side_length
    height_ratio = float(height) / output_side_length
    if model.resize_mode == 'squash' or width_ratio == height_ratio:
        image = scipy.misc.imresize(image, (output_side_length, output_side_length), interp=interp)
    elif model.resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = output_side_length
            resize_width = int(round(width / height_ratio))
        else:
            resize_width = output_side_length
            resize_height = int(round(height/ width_ratio))
        image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)

        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width-output_side_length)/2.0))
            image = image[:,start:start+output_side_length]
        else:
            start = int(round((resize_height-output_side_length)/2.0))
            image = image[start:start+output_side_length,:]
    else:
        if model.resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = output_side_length
                resize_height = int(round(height / width_ratio))
                if (output_side_length - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = output_side_length
                resize_width = int(round(width/ height_ratio))
                if (output_side_length - resize_width) % 2 == 1:
                    resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
        elif model.resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(width / new_ratio))
            resize_height = int(round(height / new_ratio))
            if width_ratio > height_ratio and (output_side_length - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (output_side_length - resize_width) % 2 == 1:
                resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width-output_side_length)/2.0))
                image = image[:,start:start+output_side_length]
            else:
                start = int(round((resize_height-output_side_length)/2.0))
                image = image[start:start+output_side_length,:]
        else:
            raise Exception('unrecognized resize_mode "%s"' % model.resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = int((output_side_length - resize_height)/2)
            noise_size = (padding, output_side_length)
            if model.channels > 1:
                noise_size += (model.channels,)
            noise = numpy.random.randint(0, 255, noise_size).astype('uint8')
            image = numpy.concatenate((noise, image, noise), axis=0)
        else:
            padding = int((output_side_length - resize_width)/2)
            noise_size = (output_side_length, padding)
            if model.channels > 1:
                noise_size += (model.channels,)
            noise = numpy.random.randint(0, 255, noise_size).astype('uint8')
            image = numpy.concatenate((noise, image, noise), axis=1)
    cv2.imwrite(dest, image)
    return

def compute_mean(data_path):
    sum_image = None
    count = 0
    train_text = os.path.join(data_path, 'train.txt')
    for line in open(train_text):
        filepath = line.strip().split()[0]
        image = numpy.asarray(Image.open(filepath))
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        else:
            zeros = numpy.zeros((256, 256))
            image = numpy.array([image, zeros, zeros])
        if sum_image is None:
            sum_image = numpy.ndarray(image.shape, dtype=numpy.float32)
            sum_image[:] = image
        else:
            sum_image += image
        count += 1
    mean = sum_image / count
    pickle.dump(mean, open(os.path.join(data_path, 'mean.npy'), 'wb'), -1)