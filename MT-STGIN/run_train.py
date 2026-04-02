# -- coding: utf-8 --
"""
the shape of sparsetensor is a tuple, like this
(array([[  0, 297],
       [  0, 296],
       ...
       [161,   0]], dtype=int32), array([0.00323625, ...], dtype=float32), (162, 300))
axis=0: is nonzero values, x-axis represents Row, y-axis represents Column.
axis=1: corresponding the nonzero value.
axis=2: represents the sparse matrix shape.
"""

from __future__ import division
from __future__ import print_function
from models.utils import *
from models.inits import *
from models.models import GCN
from models.hyparameter import parameter
from models.embedding import embedding
from models.encoder import Encoder_ST
from models.decoder import Decoder_ST
from models.bridge import BridgeTransformer
from models.bridge_lstm import LstmClass
from models.inference import InferenceClass
from models.data_next import DataClass
from models.bridge import transformAttention

import os
import datetime
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Model(tf.Module):
    def __init__(self, para):
        super().__init__(name='mt_stgin')
        self.para = para
        self.adj = preprocess_adj(self.adjecent())

        # define gcn model
        if self.para.model_name == "gcn_cheby":
            tuples = chebyshev_polynomials(self.adj, self.para.max_degree)
            self.support = [tf.SparseTensor(indices=t[0], values=tf.cast(t[1], tf.float32), dense_shape=t[2]) for t in tuples]
            self.num_supports = 1 + self.para.max_degree
            self.model_func = GCN
        else:
            self.support = [tf.SparseTensor(indices=self.adj[0], values=tf.cast(self.adj[1], tf.float32), dense_shape=self.adj[2])]
            self.num_supports = 1
            self.model_func = GCN

        # Placeholders dict (used as a container for passing dropout etc.)
        self.placeholders = {
            "dropout": 0.0,
            "num_features_nonzero": 0,
        }

        # Pre-create Dense/Conv1D layers for the model
        self._build_layers()

        # Dummy eager pass to initialize all variable shapes before @tf.function traces
        # This prevents Keras layers from complaining about being built the first time in a trace.
        _dummy_xs = tf.zeros([self.para.batch_size, self.para.input_length, self.para.site_num, self.para.features], dtype=tf.float32)
        _dummy_1d = tf.zeros([self.para.batch_size * (self.para.input_length + self.para.output_length), self.para.site_num], dtype=tf.int32)
        self.forward(_dummy_xs, _dummy_1d, _dummy_1d, _dummy_1d, _dummy_1d)

    def adjecent(self):
        """
        :return: adj matrix
        """
        data = pd.read_csv(filepath_or_buffer=self.para.file_adj)
        adj = np.zeros(shape=[self.para.site_num, self.para.site_num])
        for line in data[["src_FID", "nbr_FID"]].values:
            adj[line[0]][line[1]] = 1
        return adj

    def _build_layers(self):
        """Pre-create all layers."""
        # Embedding layers
        init = tf.initializers.TruncatedNormal(stddev=1, seed=0)
        self.pos_emb_layer = tf.keras.layers.Embedding(self.para.site_num, self.para.emb_size, embeddings_initializer=init, name="position_embed")
        self.day_emb_layer = tf.keras.layers.Embedding(32, self.para.emb_size, embeddings_initializer=init, name="day_embed")
        self.dow_emb_layer = tf.keras.layers.Embedding(8, self.para.emb_size, embeddings_initializer=init, name="day_of_week_embed")
        self.hour_emb_layer = tf.keras.layers.Embedding(24, self.para.emb_size, embeddings_initializer=init, name="hour_embed")
        self.minute_emb_layer = tf.keras.layers.Embedding(4, self.para.emb_size, embeddings_initializer=init, name="minute_embed")

        # Conv1D layers for encoder speed processing
        if self.para.model_name != "MT-STGIN-1":
            self.conv1 = tf.keras.layers.Conv1D(
                filters=self.para.emb_size, kernel_size=2, padding="same",
                kernel_initializer=tf.initializers.TruncatedNormal(),
                name="conv_1")
            self.conv2 = tf.keras.layers.Conv1D(
                filters=self.para.emb_size, kernel_size=3, padding="same",
                kernel_initializer=tf.initializers.TruncatedNormal(),
                name="conv_2")
            self.conv3 = tf.keras.layers.Conv1D(
                filters=self.para.emb_size, kernel_size=1, padding="same",
                kernel_initializer=tf.initializers.TruncatedNormal(),
                name="conv_3")

        # Inference module
        self.inference_module = InferenceClass(para=self.para)

        # ST Embedding module
        self.st_embedding_layer = STEmbeddingLayer(D=self.para.emb_size)

        # Pre-create complex network components
        self.encoder = Encoder_ST(hp=self.para, placeholders=self.placeholders, model_func=self.model_func, supports=self.support)
        self.decoder = Decoder_ST(hp=self.para, placeholders=self.placeholders, model_func=self.model_func, supports=self.support)
        self.bridge = BridgeTransformer(self.para)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.para.learning_rate)

    def forward(self, features_s, day, d_o_w, hour, minute):
        """
        Forward pass of the model.

        :param features_s: [batch, input_length, site_num, features]
        :param day: [batch*(input+output), site_num]
        :param d_o_w: [batch*(input+output), site_num]
        :param hour: [batch*(input+output), site_num]
        :param minute: [batch*(input+output), site_num]
        :return: predictions [batch, site_num, output_length]
        """
        position = np.array([[i for i in range(self.para.site_num)]], dtype=np.int32)

        # Create SparseTensors for GCN supports
        supports = [
            tf.SparseTensor(
                indices=tf.cast(self.adj[0], tf.int64),
                values=tf.cast(self.adj[1], tf.float32),
                dense_shape=tf.cast(self.adj[2], tf.int64),
            )
            for _ in range(self.num_supports)
        ]

        # Embeddings
        p_emd = self.pos_emb_layer(position)
        p_emd = tf.reshape(p_emd, shape=[1, self.para.site_num, self.para.emb_size])
        p_emd = tf.tile(
            tf.expand_dims(p_emd, axis=0),
            [self.para.batch_size, self.para.input_length + self.para.output_length, 1, 1])

        d_emb = self.day_emb_layer(day)
        d_emd = tf.reshape(d_emb, shape=[self.para.batch_size,
                                          self.para.input_length + self.para.output_length,
                                          self.para.site_num, self.para.emb_size])

        d_o_w_emb = self.dow_emb_layer(d_o_w)
        d_o_w_emd = tf.reshape(d_o_w_emb, shape=[self.para.batch_size,
                                                   self.para.input_length + self.para.output_length,
                                                   self.para.site_num, self.para.emb_size])

        h_emb = self.hour_emb_layer(hour)
        h_emd = tf.reshape(h_emb, shape=[self.para.batch_size,
                                          self.para.input_length + self.para.output_length,
                                          self.para.site_num, self.para.emb_size])

        m_emb = self.minute_emb_layer(minute)
        m_emd = tf.reshape(m_emb, shape=[self.para.batch_size,
                                          self.para.input_length + self.para.output_length,
                                          self.para.site_num, self.para.emb_size])

        # Encoder
        print("#................................in the encoder step....................................#")
        if self.para.model_name == "MT-STGIN-1":
            speed = FC(features_s, units=[self.para.emb_size, self.para.emb_size],
                      activations=[tf.nn.relu, None], bn=False, bn_decay=0.99,
                      is_training=self.para.is_training)
        else:
            speed = tf.transpose(features_s, perm=[0, 2, 1, 3])
            speed = tf.reshape(speed, [-1, self.para.input_length, self.para.features])
            speed1 = self.conv1(speed)
            speed2 = self.conv2(speed)
            speed3 = self.conv3(speed)
            speed = tf.add_n([speed1, speed2, speed3])
            speed = tf.nn.relu(speed)
            speed = tf.reshape(speed, [-1, self.para.site_num, self.para.input_length, self.para.emb_size])
            speed = tf.transpose(speed, perm=[0, 2, 1, 3])

        timestamp = [h_emd]
        position_emb = p_emd

        STE = self.st_embedding_layer(position_emb, timestamp)

        encoder_outs = self.encoder.encoder_spatio_temporal(
            speed=speed, STE=STE[:, :self.para.input_length])
        print("encoder outs shape is : ", encoder_outs.shape)

        # Decoder
        print("#................................in the decoder step....................................#")
        masked_speed = tf.concat(
            [speed[:, -self.para.pre_length:],
             tf.zeros(shape=[self.para.batch_size, self.para.output_length,
                            self.para.site_num, self.para.emb_size])],
            axis=1)
        print("masked speed shape is : ", masked_speed.shape)
        decoder_outs = self.decoder.decoder_spatio_temporal(
            speed=masked_speed,
            STE=STE[:, self.para.input_length - self.para.pre_length:],
            causality=False)
        print("decoder outs shape is : ", decoder_outs.shape)

        # BridgeTrans
        print("#................................in the bridge step.....................................#")
        encoder_outs_combined = tf.concat(
            [encoder_outs, decoder_outs[:, -self.para.output_length:]], axis=1)
        bridge_outs = self.bridge.encoder(
            X=encoder_outs_combined,
            X_P=encoder_outs_combined,
            X_Q=decoder_outs[:, -self.para.output_length:],
            causality=False)
        print("bridge outs shape is : ", bridge_outs.shape)

        # Inference
        print("#...............................in the inference step...................................#")
        pres_s = self.inference_module.inference(out_hiddens=bridge_outs)
        print("predicted speeds shape is : ", pres_s.shape)

        return pres_s

    def re_current(self, a, max_val, min_val):
        return a * (max_val - min_val) + min_val

    @tf.function
    def train_step(self, x_s, day, d_o_w, hour, minute, label_s):
        with tf.GradientTape() as tape:
            pres_s = self.forward(x_s, day, d_o_w, hour, minute)
            loss = tf.reduce_mean(tf.abs(pres_s - label_s))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    @tf.function
    def eval_step(self, x_s, day, d_o_w, hour, minute):
        return self.forward(x_s, day, d_o_w, hour, minute)

    def run_epoch(self):
        """
        Training loop using tf.GradientTape.
        """
        max_mae = 100
        iterate = DataClass(self.para)

        train_dataset = iterate.get_dataset(
            batch_size=self.para.batch_size, epoch=self.para.epoch, is_training=True)

        step = 0
        for batch in train_dataset:
            x_s, day, d_o_w, hour, minute, label_s = batch

            x_s = tf.reshape(x_s, [-1, self.para.input_length, self.para.site_num, self.para.features])
            day = tf.reshape(day, [-1, self.para.site_num])
            d_o_w = tf.reshape(d_o_w, [-1, self.para.site_num])
            hour = tf.reshape(hour, [-1, self.para.site_num])
            minute = tf.reshape(minute, [-1, self.para.site_num])

            # Check batch size matches expected
            actual_batch = x_s.shape[0]
            if actual_batch != self.para.batch_size:
                continue  # skip incomplete batches

            self.placeholders["dropout"] = self.para.dropout

            loss = self.train_step(x_s, day, d_o_w, hour, minute, label_s)

            if step % 10 == 0:
                print("after %d steps, the training average loss value is : %.6f" % (step, loss.numpy()))

            # validate processing
            if step % 100 == 0 and step > 0:
                mae = self.evaluate()
                if max_mae > mae:
                    print("the validate average loss value is : %.6f" % (mae))
                    max_mae = mae
                    # Save checkpoint
                    checkpoint = tf.train.Checkpoint(model=self)
                    checkpoint.save(file_prefix=os.path.join(self.para.save_path, "ckpt"))

            step += 1

    def evaluate(self):
        """
        Evaluation loop.
        :return: mae
        """
        label_s_list, pre_s_list = list(), list()

        if not self.para.is_training:
            # Load checkpoint
            checkpoint = tf.train.Checkpoint(model=self)
            latest = tf.train.latest_checkpoint(self.para.save_path)
            if latest:
                print("the model weights has been loaded:")
                checkpoint.restore(latest)

        iterate_test = DataClass(hp=self.para)
        test_dataset = iterate_test.get_dataset(
            batch_size=self.para.batch_size, epoch=1, is_training=False)
        max_s, min_s = iterate_test.max_s["speed"], iterate_test.min_s["speed"]

        for batch in test_dataset:
            x_s, day, d_o_w, hour, minute, label_s = batch
            x_s = tf.reshape(x_s, [-1, self.para.input_length, self.para.site_num, self.para.features])
            day = tf.reshape(day, [-1, self.para.site_num])
            d_o_w = tf.reshape(d_o_w, [-1, self.para.site_num])
            hour = tf.reshape(hour, [-1, self.para.site_num])
            minute = tf.reshape(minute, [-1, self.para.site_num])

            actual_batch = x_s.shape[0]
            if actual_batch != self.para.batch_size:
                continue

            self.placeholders["dropout"] = 0.0
            pre_s = self.eval_step(x_s, day, d_o_w, hour, minute)

            label_s_list.append(label_s.numpy())
            pre_s_list.append(pre_s.numpy())

        if not label_s_list:
            print("No valid batches for evaluation")
            return 100.0

        label_s_list = np.reshape(
            np.array(label_s_list, dtype=np.float32),
            [-1, self.para.site_num, self.para.output_length],
        ).transpose([1, 0, 2])
        pre_s_list = np.reshape(
            np.array(pre_s_list, dtype=np.float32),
            [-1, self.para.site_num, self.para.output_length],
        ).transpose([1, 0, 2])
        if self.para.normalize:
            label_s_list = self.re_current(label_s_list, max_s, min_s)
            pre_s_list = self.re_current(pre_s_list, max_s, min_s)

        print("speed prediction result")
        mae, rmse, mape, cor, r2 = metric(
            pre_s_list[:28], label_s_list[:28]
        )
        for i in range(self.para.output_length):
            print("in the %d time step, the evaluating indicator" % (i + 1))
            metric(pre_s_list[:28, :, i], label_s_list[:28, :, i])

        return mae


def main(argv=None):
    """
    :param argv:
    :return:
    """
    print(
        "#......................................beginning........................................#"
    )
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print(
        "Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively)."
    )
    val = input("please input the number : ")

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

    print(
        "#...................................finished............................................#"
    )


if __name__ == "__main__":
    main()
