# -- coding: utf-8 --
from models.inits import tf

class SemanticTransformer(object):
    def __init__(self, input_len=12, emb_size=128, site_num=66, features=1):
        self.input_len = input_len
        self.emb_size = emb_size
        self.site_num = site_num
        self.features = features

        # Pre-create Conv1D layers
        self.conv1 = tf.keras.layers.Conv1D(
            filters=self.emb_size, kernel_size=2, padding='same',
            kernel_initializer=tf.initializers.TruncatedNormal(),
            name='conv_1')
        self.conv2 = tf.keras.layers.Conv1D(
            filters=self.emb_size, kernel_size=3, padding='same',
            kernel_initializer=tf.initializers.TruncatedNormal(),
            name='conv_2')
        self.conv3 = tf.keras.layers.Conv1D(
            filters=self.emb_size, kernel_size=1, padding='same',
            kernel_initializer=tf.initializers.TruncatedNormal(),
            name='conv_3')

    def transfer(self, speed=None):
        speed = tf.reshape(speed, [-1, self.input_len, self.features])
        speed1 = self.conv1(speed)
        speed2 = self.conv2(speed)
        speed3 = self.conv3(speed)
        speed = tf.add_n([speed1, speed2, speed3])
        speed = tf.nn.relu(speed)
        speed = tf.reshape(speed, [-1, self.site_num, self.input_len, self.emb_size])
        speed = tf.transpose(speed, perm=[0, 2, 1, 3])
        return speed