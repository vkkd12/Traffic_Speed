# -- coding: utf-8 --
from models.utils import *


def normalize(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization.'''
    with tf.name_scope(scope):
        inputs_shape = inputs.shape
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def multihead_attention(queries, keys, values=None, num_units=None,
                        num_heads=8, scope="multihead_attention",
                        dropout_rate=0.0, is_training=False):
    '''Applies multihead attention.'''
    if values is None:
        values = keys

    with tf.name_scope(scope):
        if num_units is None:
            num_units = queries.shape[-1]

        Q = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(queries)
        K = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(keys)
        V = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(values)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.shape[-1] ** 0.5)
        outputs = tf.nn.softmax(outputs)
        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    return outputs


def feedforward(inputs, num_units=[2048, 512], scope="multihead_attention"):
    '''Point-wise feed forward net.'''
    with tf.name_scope(scope):
        outputs = tf.keras.layers.Conv1D(
            filters=num_units[0], kernel_size=1,
            activation=tf.nn.relu, use_bias=True)(inputs)
        outputs = tf.keras.layers.Conv1D(
            filters=num_units[1], kernel_size=1,
            activation=None, use_bias=True)(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing.'''
    K = inputs.shape[-1]
    return ((1 - epsilon) * inputs) + (epsilon / K)


class BridgeTransformer():
    def __init__(self, arg):
        self.arg = arg
        self.emb_size = self.arg.emb_size
        self.is_training = arg.is_training
        self.input_length = self.arg.input_length
        self.output_length = self.arg.output_length
        self.hidden_units = arg.emb_size
        self.batch = arg.batch_size
        self.site_num = arg.site_num

        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.dropout_rate = arg.dropout

    def encoder(self, X=None, X_Q=None, X_P=None):
        '''
        :param X: [N, input_length, site_num, emb_size]
        :param X_Q: [N, output_length, site_num, emb_size]
        :param X_P: [N, input_length, site_num, emb_size]
        :return: [N, output_length, site_num, emb_size]
        '''
        input_length = X.shape[1]
        output_length = X_Q.shape[1]
        with tf.name_scope("encoder"):
            X = tf.reshape(tf.transpose(X, [0, 2, 1, 3]), shape=[-1, input_length, self.emb_size])
            X_P = tf.reshape(tf.transpose(X_P, [0, 2, 1, 3]), shape=[-1, input_length, self.emb_size])
            X_Q = tf.reshape(tf.transpose(X_Q, [0, 2, 1, 3]), shape=[-1, output_length, self.emb_size])

            for i in range(self.num_blocks):
                with tf.name_scope(f"num_blocks_{i}"):
                    X_Q = multihead_attention(queries=X_Q,
                                              keys=X_P,
                                              values=X,
                                              num_units=self.hidden_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              is_training=self.is_training)
                    X_Q = feedforward(X_Q, num_units=[4 * self.hidden_units, self.hidden_units])
        X = tf.reshape(X_Q, shape=[-1, self.site_num, output_length, self.hidden_units])
        X = tf.transpose(X, [0, 2, 1, 3])
        print('bridge layer output, X shape is : ', X.shape)
        return X


def transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    '''
    transform attention mechanism
    '''
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