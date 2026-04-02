# -- coding: utf-8 --

import tensorflow as tf


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


def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True,
              scope="embedding"):
    '''Embeds a given tensor.'''
    with tf.name_scope(scope):
        lookup_table = tf.Variable(
            tf.initializers.TruncatedNormal(mean=0, stddev=1, seed=0)(
                shape=[vocab_size, num_units]),
            dtype=tf.float32,
            name='lookup_table')
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def multihead_attention(key_emb, que_emb, queries, keys, num_units=None,
                        num_heads=8, dropout_rate=0, is_training=True,
                        causality=False, scope="multihead_attention"):
    '''Applies multihead attention.'''
    with tf.name_scope(scope):
        if num_units is None:
            num_units = queries.shape[-1]

        Q = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(queries)
        K = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(keys)
        V = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(keys)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.shape[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(key_emb, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.band_part(diag_vals, -1, 0)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(que_emb, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks

        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        outputs += queries

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
        outputs += inputs
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing.'''
    K = inputs.shape[-1]
    return ((1 - epsilon) * inputs) + (epsilon / K)


class Transformer():
    def __init__(self, arg):
        self.is_training = arg.is_training
        self.hidden_units = arg.emb_size
        self.batch = arg.batch_size
        self.input_length = arg.input_length
        self.site_num = arg.site_num

        self.num_heads = 1
        self.num_blocks = 4
        self.dropout_rate = 0.0

    def encoder(self, speed=None, day=None, hour=None, position=None):
        '''
        :param speed: [batch, time, site num, hidden size]
        :return:
        '''
        with tf.name_scope("encoder"):
            self.en_emb = tf.reshape(speed, shape=[self.batch * self.input_length, self.site_num, self.hidden_units])
            self.enc = self.en_emb

            for i in range(self.num_blocks):
                with tf.name_scope(f"num_blocks_{i}"):
                    self.enc = multihead_attention(
                        key_emb=self.en_emb,
                        que_emb=self.en_emb,
                        queries=self.enc,
                        keys=self.enc,
                        num_units=self.hidden_units,
                        num_heads=self.num_heads,
                        dropout_rate=self.dropout_rate,
                        is_training=self.is_training,
                        causality=False)
                    self.enc = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])
                    self.enc = self.enc + self.en_emb
        print('enc shape is : ', self.enc.shape)
        return self.enc