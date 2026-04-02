# -- coding: utf-8 --
from models.inits import *

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

def multihead_attention(queries, keys, num_units=None, num_heads=8,
                        dropout_rate=0, is_training=True, causality=True,
                        scope="multihead_attention", dense_layers=None):
    '''Applies multihead attention.'''
    with tf.name_scope(scope):
        if num_units is None:
            num_units = queries.shape[-1]

        if dense_layers is not None:
            Q = dense_layers['Q'](queries)
            K = dense_layers['K'](keys)
            V = dense_layers['V'](keys)
            residual = dense_layers['res'](queries)
        else:
            Q = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(queries)
            K = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(keys)
            V = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(keys)
            residual = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(queries)

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

        outputs = tf.nn.softmax(outputs)
        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        outputs += residual
        if dense_layers is not None and 'ln' in dense_layers:
            outputs = dense_layers['ln'](outputs)
        else:
            outputs = normalize(outputs)

    return outputs


def feedforward(inputs, num_units=[2048, 512], scope="multihead_attention",
                conv_layers=None):
    '''Point-wise feed forward net.'''
    with tf.name_scope(scope):
        if conv_layers is not None:
            outputs = conv_layers['inner'](inputs)
            outputs = conv_layers['readout'](outputs)
        else:
            outputs = tf.keras.layers.Conv1D(
                filters=num_units[0], kernel_size=1,
                activation=tf.nn.relu, use_bias=True)(inputs)
            outputs = tf.keras.layers.Conv1D(
                filters=num_units[1], kernel_size=1,
                activation=None, use_bias=True)(outputs)

        outputs += inputs
        if conv_layers is not None and 'ln' in conv_layers:
            outputs = conv_layers['ln'](outputs)
        else:
            outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing.'''
    K = inputs.shape[-1]
    return ((1 - epsilon) * inputs) + (epsilon / K)


class MaskTransformer(tf.Module):
    def __init__(self, arg):
        super().__init__(name='mask_transformer')
        self.arg = arg
        self.is_training = arg.is_training
        self.hidden_units = arg.emb_size
        self.batch = arg.batch_size
        self.site_num = arg.site_num

        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.dropout_rate = arg.dropout

        # Pre-create layers for each block
        self.attn_layers = []
        self.ff_layers = []
        for i in range(self.num_blocks):
            attn = {
                'Q': tf.keras.layers.Dense(self.hidden_units, activation=tf.nn.relu,
                                           name=f'block_{i}_attn_Q'),
                'K': tf.keras.layers.Dense(self.hidden_units, activation=tf.nn.relu,
                                           name=f'block_{i}_attn_K'),
                'V': tf.keras.layers.Dense(self.hidden_units, activation=tf.nn.relu,
                                           name=f'block_{i}_attn_V'),
                'res': tf.keras.layers.Dense(self.hidden_units, activation=tf.nn.relu,
                                             name=f'block_{i}_attn_res'),
                'ln': tf.keras.layers.LayerNormalization(epsilon=1e-8, name=f'block_{i}_attn_ln'),
            }
            ff = {
                'inner': tf.keras.layers.Conv1D(
                    filters=4 * self.hidden_units, kernel_size=1,
                    activation=tf.nn.relu, use_bias=True,
                    name=f'block_{i}_ff_inner'),
                'readout': tf.keras.layers.Conv1D(
                    filters=self.hidden_units, kernel_size=1,
                    activation=None, use_bias=True,
                    name=f'block_{i}_ff_readout'),
                'ln': tf.keras.layers.LayerNormalization(epsilon=1e-8, name=f'block_{i}_ff_ln'),
            }
            self.attn_layers.append(attn)
            self.ff_layers.append(ff)

    def encoder(self, inputs=None):
        '''
        :param inputs: [batch, time, site num, hidden size]
        :return:
        '''
        input_length = inputs.shape[1]
        inputs = tf.reshape(tf.transpose(inputs, [0, 2, 1, 3]),
                           shape=[-1, input_length, self.hidden_units])
        self.enc = inputs
        ## Blocks
        for i in range(self.num_blocks):
            with tf.name_scope(f"num_blocks_{i}"):
                self.enc = multihead_attention(queries=self.enc,
                                               keys=self.enc,
                                               num_units=self.hidden_units,
                                               num_heads=self.num_heads,
                                               dropout_rate=self.dropout_rate,
                                               is_training=self.is_training,
                                               causality=True,
                                               scope='self_attention',
                                               dense_layers=self.attn_layers[i])
        self.enc = tf.reshape(self.enc, shape=[-1, self.site_num, input_length, self.hidden_units])
        self.enc = tf.transpose(self.enc, [0, 2, 1, 3])
        print('enc shape is : ', self.enc.shape)
        return self.enc