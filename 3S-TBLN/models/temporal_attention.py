# -- coding: utf-8 --
from models.inits import tf


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


def multihead_attention(args, queries, keys, num_units=None, num_heads=8,
                        scope="multihead_attention", dropout_rate=0.0,
                        is_training=False):
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

        outputs = tf.nn.softmax(outputs)
        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        outputs += tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(queries)
        outputs = normalize(outputs)

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
        outputs = normalize(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing.'''
    K = inputs.shape[-1]
    return ((1 - epsilon) * inputs) + (epsilon / K)


class TemporalTransformer():
    def __init__(self, arg):
        self.arg = arg
        self.is_training = self.arg.is_training
        self.hidden_units = self.arg.emb_size
        self.batch = self.arg.batch_size
        self.site_num = self.arg.site_num
        self.input_length = self.arg.input_length

        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.dropout_rate = arg.dropout

    def encoder(self, hiddens, hidden):
        '''
        :param hiddens: [batch, time, site num, hidden size]
        :param hidden: [batch, time, site num, hidden size]
        :return:
        '''
        with tf.name_scope("temporal_encoder"):
            enc = hiddens
            dec = hidden
            for i in range(self.num_blocks):
                with tf.name_scope(f"num_blocks_{i}"):
                    dec = multihead_attention(
                        args=self.arg,
                        queries=dec,
                        keys=enc,
                        num_units=self.hidden_units,
                        num_heads=self.num_heads,
                        dropout_rate=self.dropout_rate,
                        is_training=self.is_training)
                    dec = feedforward(dec,
                                     num_units=[4 * self.hidden_units, self.hidden_units])
        print('temporal_attention layer output, dec shape is : ', dec.shape)
        return dec