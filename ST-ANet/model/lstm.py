# -- coding: utf-8 --

import tensorflow as tf


class lstm(object):
    def __init__(self, batch_size, layer_num=1, nodes=128, is_training=True, placeholders=None):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.nodes = nodes
        self.is_training = is_training
        self.placeholders = placeholders

        # Use Keras LSTM layers
        self.lstm_layers = []
        for i in range(self.layer_num):
            self.lstm_layers.append(
                tf.keras.layers.LSTMCell(units=self.nodes, name=f'lstm_cell_{i}')
            )
        self.rnn_cell = tf.keras.layers.StackedRNNCells(self.lstm_layers)
        self.rnn_layer = tf.keras.layers.RNN(self.rnn_cell, return_sequences=True, return_state=True)

    def encoding(self, inputs=None):
        '''
        :param inputs: [batch, time, features]
        :return: (h_states, final_state)
        '''
        outputs = self.rnn_layer(inputs)
        h_states = outputs[0]  # shape: [batch, time, nodes]
        states = outputs[1:]   # final states
        return (h_states, states)