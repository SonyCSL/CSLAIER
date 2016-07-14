# -*- encoding:utf-8 -*-
# HINT:image
import tensorflow as tf

'''
Alexnet:
ImageNet Classification with Deep Convolutional Neural Networks
http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
'''


def inference(images, keep_prob):

    def weight_variable(shape):
        # conv : shape=[kernel_height, kernel_widht, network_input, network_output]
        # fc : shape = [network_input, network_output]
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=1e-1)
        return tf.Variable(initial, name='weights')

    def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
        return tf.nn.conv2d(x, W, strides, padding=padding)

    def bias_variable(shape, conv):
        initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
        biases = tf.Variable(initial, name='biases')
        bias = tf.nn.bias_add(conv, biases, 'NHWC')
        return bias

    def max_pool(input, name, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1]):
        return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='VALID', data_format='NHWC', name=name)

    x = tf.reshape(images, [-1, 128, 128, 3])

    # conv1
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([11, 11, 3, 64])
        c_conv1 = conv2d(x, W_conv1, strides=[1, 4, 4, 1], padding='VALID')
        b_conv1 = bias_variable([64], c_conv1)
        h_conv1 = tf.nn.relu(b_conv1, name=scope)

    # pool1
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool(h_conv1, scope)

    # conv2
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 64, 192])
        c_conv2 = conv2d(h_pool1, W_conv2)
        b_conv2 = bias_variable([192], c_conv2)
        h_conv2 = tf.nn.relu(b_conv2, name=scope)

    # pool2
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool(h_conv2, scope)

    # conv3
    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 192, 384])
        c_conv3 = conv2d(h_pool2, W_conv3)
        b_conv3 = bias_variable([384], c_conv3)
        h_conv3 = tf.nn.relu(b_conv3, name=scope)

    # conv4
    with tf.name_scope('conv4') as scope:
        W_conv4 = weight_variable([3, 3, 384, 256])
        c_conv4 = conv2d(h_conv3, W_conv4)
        b_conv4 = bias_variable([256], c_conv4)
        h_conv4 = tf.nn.relu(b_conv4, name=scope)

    # conv5
    with tf.name_scope('conv5') as scope:
        W_conv5 = weight_variable([3, 3, 256, 256])
        c_conv5 = conv2d(h_conv4, W_conv5)
        b_conv5 = bias_variable([256], c_conv5)
        h_conv5 = tf.nn.relu(b_conv5, name=scope)

    # pool5
    with tf.name_scope('pool5') as scope:
        h_pool5 = max_pool(h_conv5, scope)

    # fc6
    with tf.name_scope('fc6') as scope:
        r_fc6 = tf.reshape(h_pool5, [-1, 256 * 2 * 2])
        W_fc6 = weight_variable([256 * 2 * 2, 4096])
        b_fc6 = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name='biases')
        h_fc6 = tf.nn.relu_layer(r_fc6, W_fc6, b_fc6, name=scope)
        h_fc6_dropout = tf.nn.dropout(h_fc6, keep_prob)

    # fc7
    with tf.name_scope('fc7') as scope:
        W_fc7 = weight_variable([4096, 4096])
        b_fc7 = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name='biases')
        h_fc7 = tf.nn.relu_layer(h_fc6_dropout, W_fc7, b_fc7, name=scope)
        h_fc7_dropout = tf.nn.dropout(h_fc7, keep_prob)

    # fc8
    with tf.name_scope('fc8') as scope:
        W_fc8 = weight_variable([4096, 1000])
        b_fc8 = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32), name='biases')
        h_fc8 = tf.matmul(h_fc7_dropout, W_fc8) + b_fc8

    return h_fc8


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def accuracy(logits, sparse_indecies):
    labels = tf.one_hot(sparse_indecies, 1000, 1, 0)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step
