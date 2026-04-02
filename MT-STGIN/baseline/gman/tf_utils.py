# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime


def conv2d(x, output_dims, kernel_size, stride=[1, 1],
           padding='SAME', use_bias=True, activation=tf.nn.relu,
           bn=False, bn_decay=None, is_training=None):
    input_dims = x.shape[-1]
    kernel_shape = kernel_size + [input_dims, output_dims]
    kernel = tf.Variable(
        tf.initializers.GlorotUniform()(shape=kernel_shape),
        dtype=tf.float32, trainable=True, name='kernel')
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding=padding)
    if use_bias:
        bias = tf.Variable(
            tf.zeros(shape=[output_dims]),
            dtype=tf.float32, trainable=True, name='bias')
        x = tf.nn.bias_add(x, bias)
    if activation is not None:
        x = activation(x)
    return x


def batch_norm(x, is_training, bn_decay):
    input_dims = x.shape[-1]
    moment_dims = list(range(len(x.shape) - 1))
    beta = tf.Variable(
        tf.zeros(shape=[input_dims]),
        dtype=tf.float32, trainable=True, name='beta')
    gamma = tf.Variable(
        tf.ones(shape=[input_dims]),
        dtype=tf.float32, trainable=True, name='gamma')
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')
    x = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
    return x


def dropout(x, drop, is_training):
    if is_training:
        x = tf.nn.dropout(x, rate=drop)
    return x