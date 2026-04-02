# -- coding: utf-8 --
from models.inits import tf
from models.temporal_attention import TemporalTransformer


class InferenceClass(object):
    def __init__(self, para=None):
        self.para = para
        # Pre-create Dense layers
        self.dense_speed_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='layer_speed_1')
        self.dense_speed_2 = tf.keras.layers.Dense(1, activation=tf.nn.relu, name='layer_speed_2')
        self.dense_dyn_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='layer_dyn_1')
        self.dense_dyn_2 = tf.keras.layers.Dense(1, activation=tf.nn.relu, name='layer_dyn_2')

    def weighs_add(self, inputs, hidden_size):
        u_context = tf.Variable(tf.random.truncated_normal([hidden_size]), name='u_context')
        h = tf.keras.layers.Dense(hidden_size)(inputs)
        alpha = tf.nn.softmax(tf.reduce_sum(h, axis=2, keepdims=True), axis=1)
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output

    def cnn(self, x=None):
        '''
        :param x: [N, output_length, site_num, emb_size]
        :return:
        '''
        filter1 = tf.Variable(initial_value=tf.random.normal(shape=[3, 108, self.para.emb_size, 64]), name='fitter_1')
        layer1 = tf.nn.conv2d(input=x, filters=filter1, strides=[1, 1, 1, 1], padding='SAME')
        layer1 = tf.nn.sigmoid(layer1)
        print('layer1 shape is : ', layer1.shape)

        layer3 = tf.reduce_mean(layer1, axis=2)

        results_pollution = tf.keras.layers.Dense(64, name='layer_pollution_1')(layer3)
        results_pollution = tf.keras.layers.Dense(1, name='layer_pollution_2')(results_pollution)
        results_pollution = tf.squeeze(results_pollution, axis=-1)
        return results_pollution

    def inference(self, out_hiddens):
        '''
        :param out_hiddens: [N, output_length, site_num, emb_size]
        :return:
        '''
        results_speed = self.dense_speed_1(tf.transpose(out_hiddens, [0, 2, 1, 3]))
        results_speed = self.dense_speed_2(results_speed)
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
        for i in range(self.para.output_length):
            pre_features = STE[:, i:i+1, :, :]
            pre_features = tf.reshape(tf.transpose(pre_features, perm=[0, 2, 1, 3]),
                                      shape=[-1, 1, self.para.emb_size])

            print('in the decoder step, the input_features shape is : ', features.shape)
            print('in the decoder step, the pre_features shape is : ', pre_features.shape)
            T = TemporalTransformer(arg=self.para)
            x_t = T.encoder(hiddens=features, hidden=pre_features)
            x = tf.squeeze(x_t, axis=1)
            x = tf.reshape(x, shape=[-1, self.para.site_num, self.para.emb_size])
            x = self.dense_dyn_1(x)
            pre = self.dense_dyn_2(x)
            pres.append(pre)

        return tf.concat(pres, axis=-1)