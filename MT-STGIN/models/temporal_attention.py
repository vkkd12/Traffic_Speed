# -- coding: utf-8 --
from models.inits import *

def normalize(inputs,
              epsilon=1e-8,
              scope="ln"):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for name_scope.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.name_scope(scope):
        inputs_shape = inputs.shape
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        scope="multihead_attention",
                        dropout_rate=0.0,
                        is_training=False, causality=False,
                        dense_layers=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `name_scope`.
      dense_layers: Dict of pre-created Dense layers for weight reuse.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.name_scope(scope):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.shape[-1]

        # Linear projections
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

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.shape[-1] ** 0.5)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.band_part(diag_vals, -1, 0)  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += residual

        # Normalize
        if dense_layers is not None and 'ln' in dense_layers:
            outputs = dense_layers['ln'](outputs)
        else:
            outputs = normalize(outputs) # (N, T_q, C)

    return outputs


def feedforward(inputs, num_units=[2048, 512], scope="multihead_attention",
                conv_layers=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `name_scope`.
      conv_layers: Dict of pre-created Conv1D layers for weight reuse.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.name_scope(scope):
        if conv_layers is not None:
            outputs = conv_layers['inner'](inputs)
            outputs = conv_layers['readout'](outputs)
        else:
            # Inner layer
            outputs = tf.keras.layers.Conv1D(
                filters=num_units[0], kernel_size=1,
                activation=tf.nn.relu, use_bias=True)(inputs)
            # Readout layer
            outputs = tf.keras.layers.Conv1D(
                filters=num_units[1], kernel_size=1,
                activation=None, use_bias=True)(outputs)

        # Residual connection
        outputs += inputs

        # Normalize
        if conv_layers is not None and 'ln' in conv_layers:
            outputs = conv_layers['ln'](outputs)
        else:
            outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.'''
    K = inputs.shape[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


class TemporalTransformer(tf.Module):
    def __init__(self, arg):
        super().__init__(name='temporal_transformer')
        self.arg = arg
        self.is_training = self.arg.is_training
        self.hidden_units = self.arg.emb_size
        self.batch = self.arg.batch_size
        self.site_num = self.arg.site_num
        self.input_length = self.arg.input_length

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

    def encoder(self, hiddens, hidden, causality=False):
        '''
        :param hiddens: [batch, time, hidden size]
        :param hidden: [batch, time, hidden size]
        :return:
        '''
        with tf.name_scope("temporal_encoder"):
            enc = hiddens
            dec = hidden
            ## Blocks
            for i in range(self.num_blocks):
                with tf.name_scope(f"num_blocks_{i}"):
                    ### Multihead Attention
                    dec = multihead_attention(queries=dec,
                                              keys=enc,
                                              num_units=self.hidden_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              is_training=self.is_training,
                                              causality=causality,
                                              dense_layers=self.attn_layers[i])
                    ### Feed Forward
                    dec = feedforward(dec,
                                     num_units=[4 * self.hidden_units, self.hidden_units],
                                     conv_layers=self.ff_layers[i])
        return dec