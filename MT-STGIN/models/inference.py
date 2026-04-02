# -- coding: utf-8 --
from models.inits import *
from models.temporal_attention import TemporalTransformer

class InferenceClass(object):
    def __init__(self, para=None):
        self.para = para
        # Pre-create Dense layers
        self._dense_layers_created = False

    def _create_layers(self):
        if not self._dense_layers_created:
            if self.para.model_name == 'MT-STGIN-5':
                self.dense_speed_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='layer_speed_1')
                self.dense_speed_2 = tf.keras.layers.Dense(1, activation=tf.nn.relu, name='layer_speed_2')
            else:
                self.dense_task_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='task_1')
                self.dense_task_2 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='task_2')
                self.dense_task_3 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='task_3')
                self.dense_task_1_1 = tf.keras.layers.Dense(1, name='task_1_1')
                self.dense_task_2_1 = tf.keras.layers.Dense(1, name='task_2_1')
                self.dense_task_3_1 = tf.keras.layers.Dense(1, name='task_3_1')
            self._dense_layers_created = True

    def weighs_add(self, inputs, hidden_size):
        u_context = tf.Variable(tf.random.truncated_normal([hidden_size]), name='u_context')
        h = tf.keras.layers.Dense(hidden_size)(inputs)
        alpha = tf.nn.softmax(tf.reduce_sum(h, axis=2, keepdims=True), axis=1)
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output

    def inference(self, out_hiddens):
        '''
        :param out_hiddens: [N, output_length, site_num, emb_size]
        :return:
        '''
        self._create_layers()

        if self.para.model_name == 'MT-STGIN-5':
            results_speed = self.dense_speed_1(tf.transpose(out_hiddens, [0, 2, 1, 3]))
            results_speed = self.dense_speed_2(results_speed)
            results_speed = tf.squeeze(results_speed, axis=-1)
        else:
            results_1 = self.dense_task_1(out_hiddens[:, 0:28])
            results_2 = self.dense_task_2(out_hiddens[:, 28:52])
            results_3 = self.dense_task_3(out_hiddens[:, 52:])

            results_1 = self.dense_task_1_1(results_1)
            results_2 = self.dense_task_2_1(results_2)
            results_3 = self.dense_task_3_1(results_3)
            results_speed = tf.concat([results_1, results_2, results_3], axis=1)
            results_speed = tf.transpose(results_speed, [0, 2, 1, 3])
            results_speed = tf.squeeze(results_speed, axis=-1)

        return results_speed  # [N, site_num, output_length]

    def dynamic_inference(self, features=None, STE=None):
        '''
        :param features: [N, output_length, site_num, emb_size]
        :return:
        '''
        pres = list()
        features = tf.reshape(tf.transpose(features, perm=[0, 2, 1, 3]),
                              shape=[-1, self.para.input_length, self.para.emb_size])

        dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='dyn_layer_1')
        dense_2 = tf.keras.layers.Dense(1, activation=tf.nn.relu, name='dyn_layer_2')

        for i in range(self.para.output_length):
            pre_features = STE[:, i:i+1, :, :]
            pre_features = tf.reshape(tf.transpose(pre_features, perm=[0, 2, 1, 3]),
                                      shape=[-1, 1, self.para.emb_size])

            print('in the decoder step, the input_features shape is : ', features.shape)
            print('in the decoder step, the pre_features shape is : ', pre_features.shape)
            T = TemporalTransformer(arg=self.para)
            x_t = T.encoder(hiddens=features,
                            hidden=pre_features)
            x = tf.squeeze(x_t, axis=1)
            x = tf.reshape(x, shape=[-1, self.para.site_num, self.para.emb_size])
            x = dense_1(x)
            pre = dense_2(x)
            pres.append(pre)

        return tf.concat(pres, axis=-1)