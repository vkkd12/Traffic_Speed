# -- coding: utf-8 --
from models.inits import *

def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=False,
              scale=False,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean (unused in TF2, kept for API compatibility).
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    '''
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