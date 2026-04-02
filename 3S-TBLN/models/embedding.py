# -- coding: utf-8 --
from models.inits import tf

def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=False,
              scale=False,
              scope="embedding",
              reuse=None):
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