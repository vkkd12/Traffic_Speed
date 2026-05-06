# -- coding: utf-8 --

import tensorflow as tf


class lstm(object):
    def __init__(
        self, batch_size, layer_num=1, nodes=128, is_training=True, placeholders=None
    ):
        """
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        """
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.nodes = nodes
        self.is_training = is_training
        self.placeholders = placeholders

        # Use stacked LSTM layers (TF 2.x compatible)
        self.lstm_layers = []
        for i in range(self.layer_num):
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    units=self.nodes,
                    return_sequences=True,
                    return_state=True,
                    name=f"lstm_layer_{i}",
                )
            )

    def encoding(self, inputs=None):
        """
        :param inputs: [batch, time, features]
        :return: (h_states, final_state)
        """
        x = inputs
        h_states_list = []

        # Pass through stacked LSTM layers
        for lstm_layer in self.lstm_layers:
            outputs = lstm_layer(x)
            h_states = outputs[0]  # [batch, time, nodes]
            h_state_final = outputs[1]  # [batch, nodes]
            c_state_final = outputs[2]  # [batch, nodes]

            h_states_list.append(h_state_final)
            h_states_list.append(c_state_final)
            x = h_states  # Use output for next layer

        return (x, tuple(h_states_list))  # Return sequences and final states
