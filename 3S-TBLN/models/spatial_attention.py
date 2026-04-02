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
                        dropout_rate=0, is_training=True, causality=False,
                        scope="multihead_attention"):
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

        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.band_part(diag_vals, -1, 0)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        # top-k spatial attention
        values, _ = tf.math.top_k(input=outputs, k=args.spatial_top_k)
        min_ = tf.reduce_min(values, axis=-1, keepdims=True)
        outputs = tf.where(tf.math.greater(outputs, min_), outputs, tf.ones_like(outputs) * (-2**32 + 1))
        outputs = tf.nn.softmax(outputs)

        st_weights = outputs

        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        outputs += tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(queries)
        outputs = normalize(outputs)

    return outputs, st_weights


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


class SpatialTransformer():
    def __init__(self, arg):
        self.arg = arg
        self.is_training = arg.is_training
        self.hidden_units = arg.emb_size
        self.batch = arg.batch_size
        self.site_num = arg.site_num

        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.dropout_rate = arg.dropout

    def encoder(self, X=None, Y=None):
        '''
        :param X: [batch, time, site num, hidden size]
        :param Y: [batch, time, site num, hidden size]
        :return:
        '''
        with tf.name_scope("encoder"):
            self.enc = X
            self.dec = Y
            self.st_weights = list()
            for i in range(self.num_blocks):
                with tf.name_scope(f"num_blocks_{i}"):
                    self.dec, st_weights = multihead_attention(
                        args=self.arg,
                        queries=self.dec,
                        keys=self.enc,
                        num_units=self.hidden_units,
                        num_heads=self.num_heads,
                        dropout_rate=self.dropout_rate,
                        is_training=self.is_training,
                        causality=False)
                    self.dec = feedforward(self.dec,
                                         num_units=[4 * self.hidden_units, self.hidden_units])
                    self.st_weights.append(st_weights)
        print('dec shape is : ', self.dec.shape)
        return self.dec