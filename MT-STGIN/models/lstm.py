# -- coding: utf-8 --

from models.inits import *

class LstmClass(object):
    def __init__(self, batch_size, layer_num=1, nodes=128, placeholders=None):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        '''
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.nodes = nodes
        self.placeholders = placeholders
        self.encoder_init()

    def encoder_init(self):
        '''
        :return:
        '''
        # Build stacked LSTM using Keras
        cells = [tf.keras.layers.LSTMCell(self.nodes) for _ in range(self.layer_num)]
        self.stacked_cell = tf.keras.layers.StackedRNNCells(cells)
        self.rnn_layer = tf.keras.layers.RNN(self.stacked_cell, return_sequences=True, return_state=True)

    def encoding(self, inputs=None):
        '''
        :param inputs:
        :return:
        '''
        results = self.rnn_layer(inputs)
        h_states = results[0]  # output sequences
        c_states = results[1:]  # final states
        return (h_states, c_states)