# -- coding: utf-8 --
from models.inits import *


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def sparse_dropout(x, keep_prob, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False, dim=64):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """

    if sparse:
        site_num = y.shape[1]
        y = tf.transpose(y, perm=[1, 2, 0])
        y = tf.reshape(y, shape=[site_num, -1])
        res = tf.sparse.sparse_dense_matmul(x, y)
        y = tf.reshape(res, shape=[site_num, dim, -1])
        res = tf.transpose(y, perm=[2, 0, 1])
    else:
        shape = x.shape.as_list()  # [-1, site num, hidden size]
        x = tf.reshape(x, shape=[-1, shape[2]])
        res = tf.matmul(x, y)
        res = tf.reshape(res, shape=[-1, shape[1], y.shape[1]])
    return res


class GraphConvolution(tf.Module):
    """
    Graph convolution layer.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 placeholders,
                 supports,
                 dropout=0.,
                 sparse_inputs=False,
                 act=tf.nn.relu,
                 bias=False,
                 featureless=False,
                 res_name='layer'):

        super().__init__(name='graphconvolution')
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.vars = {}
        self.act = act
        self.support = supports
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_name = res_name

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        for i in range(len(self.support)):
            self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(i))
        if self.bias:
            self.vars['bias'] = zeros([output_dim], name='bias')

        # Create the residual Dense layer
        self.res_dense = tf.keras.layers.Dense(output_dim, name=self.res_name)

    def forward(self, inputs):
        x = inputs

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs, dim=self.input_dim)
            else:
                pre_sup = self.vars['weights_' + str(i)]

            support = dot(self.support[i], pre_sup, sparse=True, dim=self.output_dim)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        # residual connection layer
        res_c = self.res_dense(inputs)

        return tf.add(x=self.act(output), y=res_c)