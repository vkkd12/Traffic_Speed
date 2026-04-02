# -- coding: utf-8 --
'''
tf.keras.layers.Conv1D is the TF2 replacement for tf.layers.conv1d
'''
import tensorflow as tf

conv1d = tf.keras.layers.Conv1D


def attn_head(seq=None, out_sz=None, bias_mat=None, activation=tf.nn.elu, in_drop=0.0, coef_drop=0.0, residual=False):
    '''
    self attention
    :param seq: shape is [batch_size, seq_length, embedding_dim]
    :param out_sz:
    :param bias_mat:
    :param activation:
    :param in_drop:   dropout
    :param coef_drop: dropout
    :param residual:
    :return:
    '''

    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, rate=in_drop)

        seq_fts = tf.keras.layers.Conv1D(out_sz, 1, use_bias=False)(seq)

        # simplest self-attention possible
        f_1 = tf.keras.layers.Conv1D(1, 1)(seq_fts)
        f_2 = tf.keras.layers.Conv1D(1, 1)(seq_fts)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])

        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, rate=coef_drop)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, rate=in_drop)

        vals = tf.matmul(coefs, seq_fts)

        # Manual bias add (replaces tf.contrib.layers.bias_add)
        bias = tf.Variable(tf.zeros([vals.shape[-1]]), name='attn_bias')
        ret = tf.nn.bias_add(vals, bias)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.keras.layers.Conv1D(ret.shape[-1], 1)(seq)
            else:
                ret = ret + seq

        return activation(ret)  # activation


if __name__ == '__main__':
    x = tf.random.normal(shape=[32, 16, 300])
    att = attn_head(seq=x, out_sz=20)
