# -- coding: utf-8 --
from models.utils import *
from models.inits import *


def channel_combine(x=None):
    '''
    :param x: [-1, channel, site, dim]
    :return: [-1, 1, site, dim * channel number]
    '''
    x = tf.concat(tf.split(x, channel, axis=1), axis=-1)
    return x


def st_attention(x, is_encoder=True, pre_x=None, channels=3, input_len=12):
    '''
    :param x:  [-1, len, site, dim]
    :return: [-1, len, site, dim * channel number]
    '''
    global channel
    channel = channels
    if channels > 1:
        x = tf.concat([x[:, -channels:], x], axis=1)
    x = tf.concat(list(map(channel_combine, [x[:, i:i + channels] for i in range(input_len)])), axis=1)
    return x


def siteCombine_tf(x=None, channels=3):
    '''
    :param x: [-1, channel, site, dim]
    :return: [-1, 1, site * channel number, dim]
    '''
    x = tf.concat(tf.split(x, channels, axis=1), axis=2)
    return x


def STHolistic_tf(x, is_encoder=True, pre_x=None, channels=3, input_len=12):
    '''
    :param x: [-1, len, site, dim]
    :return: [-1, len, site * channel number, dim]
    '''
    if channels > 1 and is_encoder:
        x = tf.concat([x[:, -channels:], x], axis=1)
    elif channels > 1:
        x = tf.concat([pre_x, x], axis=1)
    x = tf.concat([siteCombine_tf(x[:, i:i + channels], channels=channels) for i in range(input_len)], axis=1)
    return x


def fusionGate(x, y):
    '''
    :param x: [-1, len, site, dim]
    :param y: [-1, len, site, dim]
    :return: [-1, len, site, dim]
    '''
    z = tf.nn.sigmoid(tf.multiply(x, y))
    h = tf.add(tf.multiply(z, x), tf.multiply(1 - z, y))
    return h


def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True, drop=None):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = dropout(x, drop=drop, is_training=is_training)
        x = conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x


def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    '''
    spatio-temporal embedding
    '''
    SE = FC(
        SE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    TE = tf.concat(TE, axis=-1)
    TE = FC(
        TE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return tf.add(SE, TE)


def spatialAttention(X, Key, K, d, bn, bn_decay, is_training, top_k=32):
    '''
    spatial attention mechanism
    '''
    D = K * d
    query = FC(X, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(Key, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(Key, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    attention = tf.matmul(query, key, transpose_b=True)
    attention /= (d ** 0.5)

    values, indices = tf.math.top_k(input=attention, k=top_k)
    min_ = tf.reduce_min(values, axis=-1, keepdims=True)
    attention = tf.where(tf.math.greater_equal(attention, min_), attention, tf.ones_like(attention) * (-2 ** 32 + 1))

    attention = tf.nn.softmax(attention, axis=-1)
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None], bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def temporalAttention(X, K, d, bn, bn_decay, is_training, mask=True):
    '''
    temporal attention mechanism
    '''
    D = K * d
    query = FC(X, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(X, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(X, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.shape[1]
        N = X.shape[2]
        mask_mat = tf.ones(shape=(num_step, num_step))
        mask_mat = tf.linalg.band_part(mask_mat, -1, 0)
        mask_mat = tf.expand_dims(tf.expand_dims(mask_mat, axis=0), axis=0)
        mask_mat = tf.tile(mask_mat, multiples=(K * batch_size, N, 1, 1))
        mask_mat = tf.cast(mask_mat, dtype=tf.bool)
        attention = tf.where(condition=mask_mat, x=attention, y=-2 ** 15 + 1)
    attention = tf.nn.softmax(attention, axis=-1)
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None], bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''gated fusion'''
    XS = FC(HS, units=D, activations=None, bn=bn, bn_decay=bn_decay, is_training=is_training, use_bias=False)
    XT = FC(HT, units=D, activations=None, bn=bn, bn_decay=bn_decay, is_training=is_training, use_bias=True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(H, units=[D, D], activations=[tf.nn.relu, None], bn=bn, bn_decay=bn_decay, is_training=is_training)
    return H


def BridgeTrans(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    '''transform attention mechanism'''
    D = K * d
    query = FC(STE_Q, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(STE_P, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(X, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None], bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask=True, top_k=32, is_encoder=True, pre_x=None, N=207,
               channels=3, input_len=12):
    XH = X
    HT = temporalAttention(XH, K, d, bn, bn_decay, is_training, mask=mask)
    XS = tf.concat((X, STE), axis=-1)
    XS = STHolistic_tf(XS, is_encoder=is_encoder, pre_x=pre_x, channels=channels, input_len=input_len)
    print(XS[:, :, -N:].shape, XS.shape)
    HS = spatialAttention(XS[:, :, -N:], XS, K, d, bn, bn_decay, is_training, top_k=top_k)
    H = fusionGate(HS, HT)
    return tf.add(X, H)


def TS_TBLN(XS, XS_All, TE, SE, P, Q, T, L, K, d, bn, bn_decay, is_training, top_k=32, N=207, channels=3):
    '''
    3S-TBLN
    '''
    D = K * d
    X = FC(XS, units=[D, D], activations=[tf.nn.relu, None],
           bn=bn, bn_decay=bn_decay, is_training=is_training)
    XS = X
    X_All = FC(XS_All, units=[D, D], activations=[tf.nn.relu, None],
               bn=bn, bn_decay=bn_decay, is_training=is_training)
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, :P]
    STE_Q = STE[:, P:]

    # encoder
    for i in range(L):
        with tf.name_scope(f"encoder_num_blocks_{i}"):
            X = STAttBlock(X + X_All[:, :P], STE_P, K, d, bn, bn_decay, is_training, top_k=top_k, N=N,
                           channels=channels, input_len=P)
    print('encoder output shape is ', X.shape)

    # BridgeTrans encoder
    with tf.name_scope("BridgeTrans_Encoder_1"):
        X = BridgeTrans(X, X + STE_P, STE_Q + X_All[:, P:], K, d, bn, bn_decay, is_training)
    print('bridge output shape is ', X.shape)

    # decoder
    for i in range(L):
        with tf.name_scope(f"decoder_num_blocks_{i}"):
            X = STAttBlock(X + X_All[:, P:], STE_Q, K, d, bn, bn_decay, is_training,
                           top_k=top_k, is_encoder=False,
                           pre_x=tf.concat([XS[:, -channels:] + X_All[:, P - channels:P], STE_P[:, -channels:]],
                                           axis=-1),
                           N=N, channels=channels, input_len=Q)
    print('decoder output shape is ', X.shape)
    X_en = X

    # BridgeTrans decoder
    with tf.name_scope("BridgeTrans_Decoder_1"):
        X = BridgeTrans(X, X + STE_Q, STE_P + X_All[:, :P], K, d, bn, bn_decay, is_training)
    print('decoder bridge output shape is ', X.shape)
    X_de = X

    # inference
    X_en = FC(X_en, units=[D, 1], activations=[tf.nn.relu, None],
              bn=bn, bn_decay=bn_decay, is_training=is_training,
              use_bias=True, drop=0.1)

    X_de = FC(X_de, units=[D, 1], activations=[tf.nn.relu, None],
              bn=bn, bn_decay=bn_decay, is_training=is_training,
              use_bias=True, drop=0.1)
    X = tf.concat([X_de, X_en], axis=1)
    return tf.squeeze(X, axis=3)