# -- coding: utf-8 --
from models.inits import *
import seaborn as sns


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
        if bn:
            x = batch_norm(x, is_training=is_training, bn_decay=bn_decay)
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
    batch_mean, batch_var = tf.nn.moments(x, moment_dims)
    x = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
    return x


def dropout(x, drop, is_training):
    if is_training:
        x = tf.nn.dropout(x, rate=drop)
    return x


def siteCombine(x=None):
    '''
    :param x: [-1, channel, site, dim]
    :return: [-1, 1, site, dim * channel number]
    '''
    x = np.concatenate(np.split(x, channel, axis=1), axis=2)
    return x


def STHolistic(x, is_encoder=True, pre_x=None, channels=3, input_len=12):
    '''
    :param x: [-1, len, site, dim]
    :return: [-1, len, site * channel number, dim]
    '''
    global channel
    channel = channels
    if channels > 1 and is_encoder:
        x = np.concatenate([x[:, -channels:], x], axis=1)
    elif channels > 1:
        print(pre_x.shape, x.shape)
        x = np.concatenate([pre_x, x], axis=1)
    x = np.concatenate(list(map(siteCombine, [x[:, i:i + channels] for i in range(input_len)])), axis=1)
    return x


def construct_feed_dict(xs, xs_all, label_s, d_of_week, day, hour, minute, mask=[], placeholders=None, site=207, is_training=True):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict['position'] = np.array([[i for i in range(site)]], dtype=np.int32)
    feed_dict['labels'] = label_s
    feed_dict['week'] = d_of_week
    feed_dict['day'] = day
    feed_dict['hour'] = hour
    feed_dict['minute'] = minute
    feed_dict['features'] = xs
    feed_dict['features_all'] = xs_all
    feed_dict['random_mask'] = mask
    feed_dict['is_training'] = is_training
    return feed_dict


def mae_los(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(
        condition=tf.math.is_nan(mask), x=tf.zeros_like(mask), y=mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.where(
        condition=tf.math.is_nan(loss), x=tf.zeros_like(loss), y=loss)
    loss = tf.reduce_mean(loss)
    return loss


import matplotlib.pyplot as plt


def describe(label, predict):
    '''
    :param label:
    :param predict:
    :return:
    '''
    plt.figure()
    plt.plot(label[0:], 'b', label='actual value')
    plt.plot(predict[0:], 'r', label='predicted value')
    plt.legend()
    plt.show()


def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label.astype(np.float32))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst
    return mae, rmse, mape


def seaborn(x=None, len=12, heads=4):
    '''
    :param x:
    :return:
    '''
    i = 5
    f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=heads, ncols=3)
    sns.heatmap(x[i, :, :108], annot=False, ax=ax1[0], cbar=True, cmap='Blues')
    sns.heatmap(x[i, :, 108:216], annot=False, ax=ax1[1], cbar=True, cmap='Greens')
    sns.heatmap(x[i, :, 216:], annot=False, ax=ax1[2], cbar=True, cmap='Greys')

    sns.heatmap(x[i+12, :, :108], annot=False, ax=ax2[0], cbar=True, cmap='Blues')
    sns.heatmap(x[i+12, :, 108:216], annot=False, ax=ax2[1], cbar=True, cmap='Greens')
    sns.heatmap(x[i+12, :, 216:], annot=False, ax=ax2[2], cbar=True, cmap='Greys')

    sns.heatmap(x[i+24, :, :108], annot=False, ax=ax3[0], cbar=True, cmap='Blues')
    sns.heatmap(x[i+24, :, 108:216], annot=False, ax=ax3[1], cbar=True, cmap='Greens')
    sns.heatmap(x[i+24, :, 216:], annot=False, ax=ax3[2], cbar=True, cmap='Greys')

    sns.heatmap(x[i+36, :, :108], annot=False, ax=ax4[0], cbar=True, cmap='Blues')
    sns.heatmap(x[i+36, :, 108:216], annot=False, ax=ax4[1], cbar=True, cmap='Greens')
    sns.heatmap(x[i+36, :, 216:], annot=False, ax=ax4[2], cbar=True, cmap='Greys')

    plt.show()