# -- coding: utf-8 --
from model.t_attention import T_attention
import tensorflow as tf


class lstm(object):
    def __init__(self, batch_size, predict_time, layer_num=1, nodes=128, placeholders=None):
        '''
        :param batch_size: batch * site num
        :param layer_num:
        :param nodes:
        '''
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.nodes = nodes
        self.predict_time = predict_time
        self.placeholders = placeholders

        # Build decoder LSTM using Keras
        self.lstm_cells = []
        for i in range(self.layer_num):
            self.lstm_cells.append(
                tf.keras.layers.LSTMCell(units=self.nodes, name=f'decoder_lstm_cell_{i}')
            )
        self.rnn_cell = tf.keras.layers.StackedRNNCells(self.lstm_cells)

        # Pre-create Dense layers
        self.attention_dense = tf.keras.layers.Dense(self.nodes, activation=tf.nn.relu, name='attention_dense')
        self.output_dense = tf.keras.layers.Dense(1, name='output_layer')
        self.projection_dense = tf.keras.layers.Dense(self.nodes, name='projection_dense')

    def attention(self, h_t, encoder_hs):
        '''
        h_t: [batch, 1, h]
        encoder_hs: [batch, time, h]
        :return: [batch, h]
        '''
        scores = tf.reduce_sum(tf.multiply(encoder_hs, tf.tile(h_t, multiples=[1, encoder_hs.shape[1], 1])), 2)
        a_t = tf.nn.softmax(scores)
        a_t = tf.expand_dims(a_t, 2)
        c_t = tf.matmul(tf.transpose(encoder_hs, perm=[0, 2, 1]), a_t)
        c_t = tf.squeeze(c_t, axis=2)
        h_t = tf.squeeze(h_t, axis=1)
        h_tld = self.attention_dense(tf.concat([h_t, c_t], axis=1))
        return h_tld

    def decoding(self, encoder_hs):
        '''
        :param encoder_hs: [batch, time, hidden]
        :return:
        '''
        h = list()
        h_state = encoder_hs[:, -1, :]

        # Initialize RNN state
        state = self.rnn_cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)

        for i in range(self.predict_time):
            h_state_input = tf.expand_dims(input=h_state, axis=1)

            # Single step through RNN cell
            output, state = self.rnn_cell(h_state_input[:, 0, :], state)
            h_state_out = tf.expand_dims(output, axis=1)

            h_state = self.attention(h_t=h_state_out, encoder_hs=encoder_hs)
            results = self.output_dense(h_state)
            h.append(results)
        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h), [1, 2, 0]), axis=1)

    def gcn_decoding(self, encoder_hs, gan=None, site_num=None, day=None, hour=None, position=None):
        '''
        :param encoder_hs: [batch, time, site num, hidden size]
        :return: [batch, site num, prediction size]
        '''
        pres = list()
        shape = encoder_hs.shape
        h_states = encoder_hs[:, -1, :, :]
        encoder_hs = tf.reshape(tf.transpose(encoder_hs, perm=[0, 2, 1, 3]),
                                shape=[shape[0] * shape[2], shape[1], shape[3]])

        # Initialize RNN state
        state = self.rnn_cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)

        for i in range(self.predict_time):
            out_day = day[:, i, :, :]
            out_hour = hour[:, i, :, :]

            h_states = self.projection_dense(h_states)

            gan.input_length = 1
            x = gan.encoder(speed=h_states, day=out_day, hour=out_hour, position=position[:, -1, :, :])

            features = tf.add_n([x, position[:, -1, :, :]])
            features = tf.reshape(features, shape=[self.batch_size, 1, features.shape[-1]])

            # Single step through RNN cell
            output, state = self.rnn_cell(features[:, 0, :], state)
            h_state = tf.expand_dims(output, axis=1)

            # Temporal attention
            h_state = T_attention(hiddens=encoder_hs, hidden=h_state, hidden_units=shape[-1])
            h_states = tf.reshape(h_state, shape=[-1, site_num, self.nodes])

            results = self.output_dense(h_state)
            pre = tf.reshape(results, shape=[-1, site_num])
            pres.append(tf.expand_dims(pre, axis=-1))

        return tf.concat(pres, axis=-1)