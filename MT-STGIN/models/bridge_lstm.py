# -- coding: utf-8 --

from models.inits import *

class LstmClass(object):
    def __init__(self, batch_size, predict_time=1, layer_num=1, nodes=128, placeholders=None):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.nodes = nodes
        self.predict_time = predict_time
        self.placeholders = placeholders
        self.encoder_init()
        self.decoder_init()

    def encoder_init(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''
        cells = [tf.keras.layers.GRUCell(self.nodes) for _ in range(self.layer_num)]
        self.e_stacked_cell = tf.keras.layers.StackedRNNCells(cells)
        self.e_rnn = tf.keras.layers.RNN(self.e_stacked_cell, return_sequences=True, return_state=True)

    def decoder_init(self):
        cells = [tf.keras.layers.GRUCell(self.nodes) for _ in range(self.layer_num)]
        self.d_stacked_cell = tf.keras.layers.StackedRNNCells(cells)
        self.d_rnn = tf.keras.layers.RNN(self.d_stacked_cell, return_sequences=True, return_state=True)

    def encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        results = self.e_rnn(inputs)
        self.ouputs = results[0]
        self.state = results[1:]
        return self.ouputs

    def decoding(self, encoder_hs):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''
        pres = []
        h_state = encoder_hs[:, -1:, :]
        initial_state = None

        for i in range(self.predict_time):
            results = self.d_rnn(h_state, initial_state=initial_state)
            h_state = results[0]
            initial_state = results[1:]
            if len(initial_state) == 1:
                initial_state = initial_state[0]

            pres.append(h_state)

        return tf.concat(pres, axis=1)

import numpy as np
if __name__ == '__main__':
    train_data = np.random.random(size=[32, 3, 16])
    r = LstmClass(32, 10, 2, 128)
    x = tf.constant(train_data, dtype=tf.float32)
    hs = r.encoding(x)
    print(hs.shape)
    pre = r.decoding(hs)
    print(pre.shape)