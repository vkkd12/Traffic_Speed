# -- coding: utf-8 --

from __future__ import division
from __future__ import print_function
from model.hyparameter import parameter
from model.gan import embedding
from model.gan import Transformer
from model.utils import construct_feed_dict

import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import model.decoder as decoder
import matplotlib.pyplot as plt
import model.lstm as encoder_lstm
import model.data_load as data_load
import os
import argparse
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"


class Model(tf.Module):
    def __init__(self, para):
        super().__init__(name='st_anet')
        self.para = para

        # Pre-create layers
        self.feature_dense = tf.keras.layers.Dense(self.para.emb_size, name='feature_embed')
        self.projection_dense = tf.keras.layers.Dense(self.para.emb_size, name='projection_dense')

        # Encoder components
        self.encoder_transformer = Transformer(self.para)
        self.encoder_lstm_init = encoder_lstm.lstm(
            self.para.batch_size * self.para.site_num,
            self.para.hidden_layer,
            self.para.hidden_size,
            self.para.is_training)
        self.encoder_lstm_dep = encoder_lstm.lstm(
            self.para.batch_size * self.para.site_num,
            self.para.hidden_layer,
            self.para.hidden_size,
            self.para.is_training)

        # Decoder components
        self.decoder_transformer = Transformer(self.para)
        self.decoder_lstm_init = decoder.lstm(
            self.para.batch_size * self.para.site_num,
            self.para.output_length,
            self.para.hidden_layer,
            self.para.hidden_size)

        # Embedding Layers
        init = tf.initializers.TruncatedNormal(stddev=1, seed=0)
        self.pos_emb_layer = tf.keras.layers.Embedding(self.para.site_num, self.para.emb_size, embeddings_initializer=init, name="position_embed")
        self.day_emb_layer = tf.keras.layers.Embedding(32, self.para.emb_size, embeddings_initializer=init, name="day_embed")
        self.hour_emb_layer = tf.keras.layers.Embedding(24, self.para.emb_size, embeddings_initializer=init, name="hour_embed")

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.para.learning_rate)

        # Dummy eager pass to build variables
        _dummy_features = tf.zeros([self.para.batch_size * self.para.input_length, self.para.site_num, self.para.features], dtype=tf.float32)
        _dummy_1d = tf.zeros([self.para.batch_size * (self.para.input_length + self.para.output_length), self.para.site_num], dtype=tf.int32)
        self.forward(_dummy_features, _dummy_1d, _dummy_1d)

    def forward(self, features, day, hour):
        '''
        Forward pass.
        :param features: [batch * input_len, site_num, features]
        :param day: [batch * (input+output), site_num]
        :param hour: [batch * (input+output), site_num]
        :return: predictions [batch, site_num, output_length]
        '''
        position = np.array([[i for i in range(self.para.site_num)]], dtype=np.int32)

        # Embeddings
        p_emd = self.pos_emb_layer(position)
        p_emd = tf.reshape(p_emd, shape=[1, self.para.site_num, self.para.emb_size])
        p_emd = tf.expand_dims(p_emd, axis=0)
        p_emd = tf.tile(p_emd, [self.para.batch_size, self.para.input_length + self.para.output_length, 1, 1])

        d_emb = self.day_emb_layer(day)
        d_emd = tf.reshape(d_emb, shape=[self.para.batch_size,
                                          self.para.input_length + self.para.output_length,
                                          self.para.site_num, self.para.emb_size])

        h_emb = self.hour_emb_layer(hour)
        h_emd = tf.reshape(h_emb, shape=[self.para.batch_size,
                                          self.para.input_length + self.para.output_length,
                                          self.para.site_num, self.para.emb_size])

        # Encoder
        features_emb = self.feature_dense(features)  # [-1, site num, emb_size]
        in_day = d_emd[:, :self.para.input_length, :, :]
        in_hour = h_emd[:, :self.para.input_length, :, :]
        in_position = p_emd[:, :self.para.input_length, :, :]

        # Transformer encoder
        self.encoder_transformer.input_length = self.para.input_length
        x = self.encoder_transformer.encoder(speed=features_emb, day=in_day, hour=in_hour,
                                              position=p_emd[:, :self.para.input_length, :, :])

        x = tf.reshape(x, shape=[self.para.batch_size, self.para.input_length,
                                  self.para.site_num, self.para.emb_size])
        inputs = tf.add_n([x, in_position])
        inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
        inputs = tf.reshape(inputs, shape=[self.para.batch_size * self.para.site_num,
                                            self.para.input_length, self.para.emb_size])

        # LSTM encoder
        h_states, c_states = self.encoder_lstm_init.encoding(inputs)
        h_states = tf.reshape(h_states, shape=[self.para.batch_size, self.para.site_num,
                                                self.para.input_length, self.para.hidden_size])
        h_states = tf.transpose(h_states, perm=[0, 2, 1, 3])

        # Dependent LSTM encoder
        features_dep = tf.reshape(features_emb, [self.para.batch_size, self.para.input_length,
                                                   self.para.site_num, self.para.emb_size])
        inputs_dep = tf.transpose(features_dep, perm=[0, 2, 1, 3])
        inputs_dep = tf.reshape(inputs_dep, shape=[self.para.batch_size * self.para.site_num,
                                                     self.para.input_length, self.para.emb_size])
        h_states_dep, c_states_dep = self.encoder_lstm_dep.encoding(inputs_dep)
        h_states_dep = tf.reshape(h_states_dep, shape=[self.para.batch_size, self.para.site_num,
                                                        self.para.input_length, self.para.hidden_size])
        h_states_dep = tf.transpose(h_states_dep, perm=[0, 2, 1, 3])

        h_states = tf.add_n([h_states, h_states_dep])

        # Decoder
        out_day = d_emd[:, self.para.input_length:, :, :]
        out_hour = h_emd[:, self.para.input_length:, :, :]
        out_position = p_emd[:, self.para.input_length:, :, :]

        pres = self.decoder_lstm_init.gcn_decoding(
            h_states,
            gan=self.decoder_transformer,
            site_num=self.para.site_num,
            day=out_day,
            hour=out_hour,
            position=out_position)

        return pres

    def accuracy(self, label, predict):
        error = label - predict
        average_error = np.mean(np.fabs(error.astype(float)))
        print("mae is : %.6f" % (average_error))
        rmse_error = np.sqrt(np.mean(np.square(label - predict)))
        print("rmse is : %.6f" % (rmse_error))
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
        print('correlation coefficient is: %.6f' % (cor))
        sse = np.sum((label - predict) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        R2 = 1 - sse / sst
        print('r^2 is: %.6f' % (R2))
        return average_error, rmse_error, cor, R2

    def re_current(self, a, max_val, min_val):
        return [num * (max_val - min_val) + min_val for num in a]

    @tf.function
    def train_step(self, features, day, hour, label):
        with tf.GradientTape() as tape:
            pres = self.forward(features, day, hour)
            loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_mean(tf.square(pres + 1e-10 - label), axis=0)))

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    @tf.function
    def eval_step(self, features, day, hour):
        return self.forward(features, day, hour)

    def run_epoch(self):
        '''Training loop using tf.GradientTape.'''
        max_rmse = 100

        iterate = data_load.DataIterator(
            site_id=self.para.target_site_id,
            is_training=self.para.is_training,
            time_size=self.para.input_length,
            prediction_size=self.para.output_length,
            data_divide=self.para.data_divide,
            window_step=self.para.step,
            normalize=self.para.normalize,
            hp=self.para)

        train_dataset = iterate.next_batch(batch_size=self.para.batch_size, epochs=self.para.epochs, is_training=True)

        step = 0
        for batch in train_dataset:
            x, day, hour, label = batch
            features = tf.reshape(x, [-1, self.para.site_num, self.para.features])
            day = tf.reshape(day, [-1, self.para.site_num])
            hour = tf.reshape(hour, [-1, self.para.site_num])

            # Check batch size
            actual_batch = features.shape[0] // self.para.input_length
            if actual_batch != self.para.batch_size:
                continue

            loss = self.train_step(features, day, hour, label)

            print("after %d steps, the training average loss value is : %.6f" % (step, loss.numpy()))

            if step % 10 == 0:
                rmse_error = self.evaluate()
                if max_rmse > rmse_error:
                    print("the validate average rmse loss value is : %.6f" % (rmse_error))
                    max_rmse = rmse_error
                    checkpoint = tf.train.Checkpoint(model=self)
                    checkpoint.save(file_prefix=os.path.join(self.para.save_path, 'model'))

            step += 1

    def evaluate(self):
        '''Evaluation loop.'''
        label_list = list()
        predict_list = list()

        if not self.para.is_training:
            checkpoint = tf.train.Checkpoint(model=self)
            latest = tf.train.latest_checkpoint(self.para.save_path)
            if latest:
                print('the model weights has been loaded:')
                checkpoint.restore(latest)

        iterate_test = data_load.DataIterator(
            site_id=self.para.target_site_id,
            is_training=self.para.is_training,
            time_size=self.para.input_length,
            prediction_size=self.para.output_length,
            data_divide=self.para.data_divide,
            normalize=self.para.normalize,
            hp=self.para)

        test_dataset = iterate_test.next_batch(batch_size=self.para.batch_size, epochs=1, is_training=False)
        max_val, min_val = iterate_test.max_list[-2], iterate_test.min_list[-2]

        for batch in test_dataset:
            x, day, hour, label = batch
            features = tf.reshape(x, [-1, self.para.site_num, self.para.features])
            day = tf.reshape(day, [-1, self.para.site_num])
            hour = tf.reshape(hour, [-1, self.para.site_num])

            actual_batch = features.shape[0] // self.para.input_length
            if actual_batch != self.para.batch_size:
                continue

            pre = self.eval_step(features, day, hour)
            label_list.append(label.numpy())
            predict_list.append(pre.numpy())

        if not label_list:
            print("No valid batches for evaluation")
            return 100.0

        label_list = np.reshape(np.array(label_list, dtype=np.float32),
                                [-1, self.para.site_num, self.para.output_length]).transpose([1, 0, 2])
        predict_list = np.reshape(np.array(predict_list, dtype=np.float32),
                                  [-1, self.para.site_num, self.para.output_length]).transpose([1, 0, 2])
        if self.para.normalize:
            label_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max_val, min_val) for site_label in label_list])
            predict_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max_val, min_val) for site_label in predict_list])
        else:
            label_list = np.array([np.reshape(site_label, [-1]) for site_label in label_list])
            predict_list = np.array([np.reshape(site_label, [-1]) for site_label in predict_list])

        label_list = np.reshape(label_list, [-1])
        predict_list = np.reshape(predict_list, [-1])
        average_error, rmse_error, cor, R2 = self.accuracy(label_list, predict_list)
        return average_error


def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:
        para.is_training = True
    else:
        para.batch_size = 1
        para.is_training = False

    pre_model = Model(para)

    if int(val) == 1:
        pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()