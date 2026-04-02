# -- coding: utf-8 --
from __future__ import division
from __future__ import print_function
from models.st_block import TS_TBLN
from models.inits import *
from models.utils import *
from models.hyparameter import parameter
from models.embedding import embedding
from models.data_load import *


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"


class Model(tf.Module):
    def __init__(self, para, mean, std):
        super().__init__(name='ts_tbln')
        self.para = para
        self.mean = mean
        self.std = std
        self.input_len = self.para.input_length
        self.output_len = self.para.output_length
        self.total_len = self.input_len + self.output_len
        self.features = self.para.features
        self.batch_size = self.para.batch_size
        self.epochs = self.para.epoch
        self.site_num = self.para.site_num
        self.emb_size = self.para.emb_size
        self.is_training = self.para.is_training
        self.learning_rate = self.para.learning_rate
        self.model_name = self.para.model_name
        self.granularity = self.para.granularity
        self.decay_epoch = self.para.decay_epoch
        self.num_train = 23967

        # Setup Embedding Layers
        init = tf.initializers.TruncatedNormal(stddev=1, seed=0)
        self.pos_emb_layer = tf.keras.layers.Embedding(self.site_num, self.emb_size, embeddings_initializer=init, name="position_embed")
        self.dow_emb_layer = tf.keras.layers.Embedding(7, self.emb_size, embeddings_initializer=init, name="day_of_week_embed")
        self.day_emb_layer = tf.keras.layers.Embedding(32, self.emb_size, embeddings_initializer=init, name="day_embed")
        self.hour_emb_layer = tf.keras.layers.Embedding(24, self.emb_size, embeddings_initializer=init, name="hour_embed")
        self.minute_emb_layer = tf.keras.layers.Embedding(24 * 60 // self.granularity, self.emb_size, embeddings_initializer=init, name="minute_embed")

        # Optimizer
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def _update_learning_rate(self):
        lr = self.learning_rate * (0.7 ** (tf.cast(self.global_step, tf.float32) // (self.decay_epoch * self.num_train // self.batch_size)))
        lr = tf.maximum(lr, 1e-5)
        self.optimizer.learning_rate.assign(lr)

        # Dummy eager pass to build variables
        _dummy_xs = tf.zeros([self.batch_size, self.input_len, self.site_num, self.features], dtype=tf.float32)
        _dummy_xs_all = tf.zeros([self.batch_size, self.total_len, self.site_num, self.features], dtype=tf.float32)
        _dummy_1d = tf.zeros([self.batch_size, self.total_len, self.site_num], dtype=tf.int32)
        self.forward(_dummy_xs, _dummy_xs_all, _dummy_1d, _dummy_1d, _dummy_1d, _dummy_1d, is_training=False)

    def forward(self, xs, xs_all, d_of_week, day, hour, minute, is_training=True):
        '''
        Forward pass.
        :param xs: [batch, input_len, site_num, features]
        :param xs_all: [batch, input_len+output_len, site_num, features]
        :param d_of_week: [batch, total_len, site_num]
        :param day: [batch, total_len, site_num]
        :param hour: [batch, total_len, site_num]
        :param minute: [batch, total_len, site_num]
        :return: predictions
        '''
        position = np.array([[i for i in range(self.site_num)]], dtype=np.int32)

        # Embeddings
        p_emd = self.pos_emb_layer(position)
        p_emd = tf.reshape(p_emd, shape=[1, self.site_num, self.emb_size])
        p_emd = tf.expand_dims(p_emd, axis=0)

        w_emb = self.dow_emb_layer(d_of_week)
        w_emd = tf.reshape(w_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        d_emb = self.day_emb_layer(day)
        d_emd = tf.reshape(d_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        h_emb = self.hour_emb_layer(hour)
        h_emd = tf.reshape(h_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        m_emb = self.minute_emb_layer(minute)
        m_emd = tf.reshape(m_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        # Compute bn_decay
        step = tf.cast(self.global_step, tf.float32)
        bn_momentum = 0.5 * (0.5 ** (step // (self.decay_epoch * self.num_train // self.batch_size)))
        bn_decay = tf.minimum(0.99, 1.0 - bn_momentum)

        pre = TS_TBLN(XS=xs,
                      XS_All=xs_all,
                      TE=[w_emd, m_emd],
                      SE=p_emd,
                      P=self.input_len,
                      Q=self.output_len,
                      T=60 * 24 // self.granularity,
                      L=self.para.num_blocks,
                      K=self.para.num_heads,
                      d=self.emb_size // self.para.num_heads,
                      bn=True,
                      bn_decay=bn_decay,
                      is_training=is_training,
                      top_k=self.para.spatial_top_k,
                      N=self.site_num,
                      channels=self.para.channels)
        pre = pre * self.std + self.mean
        pre = tf.transpose(pre, [0, 2, 1])
        return pre

    @tf.function
    def train_step(self, xs, xs_all, d_of_week, day, hour, minute, labels):
        self._update_learning_rate()
        
        # Loss 1: reconstruction loss
        with tf.GradientTape() as tape:
            predicted = self.forward(xs, xs_all, d_of_week, day, hour, minute, is_training=True)
            l1 = mae_los(predicted[:, :, :self.input_len], labels[:, :, :self.input_len])
        gradients_1 = tape.gradient(l1, self.trainable_variables)
        gradients_1 = [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients_1, self.trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients_1, self.trainable_variables))

        # Loss 2: prediction loss
        with tf.GradientTape() as tape:
            predicted = self.forward(xs, xs_all, d_of_week, day, hour, minute, is_training=True)
            l2 = mae_los(predicted[:, :, self.input_len:], labels[:, :, self.input_len:])
        gradients_2 = tape.gradient(l2, self.trainable_variables)
        gradients_2 = [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients_2, self.trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients_2, self.trainable_variables))

        self.global_step.assign_add(1)
        return l1, l2

    @tf.function
    def eval_step(self, xs, xs_all, d_of_week, day, hour, minute):
        return self.forward(xs, xs_all, d_of_week, day, hour, minute, is_training=False)

    def run_epoch(self, trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll,
                  valX, valDoW, valD, valH, valM, valL, valXAll,
                  testX, testDoW, testD, testH, testM, testL, testXAll):
        max_mae = 100
        shape = trainX.shape
        num_batch = math.ceil(shape[0] / self.batch_size)
        self.num_train = shape[0]

        start_time = datetime.datetime.now()
        iteration = 0

        for epoch in range(self.epochs):
            # shuffle
            permutation = np.random.permutation(shape[0])
            trainX = trainX[permutation]
            trainDoW = trainDoW[permutation]
            trainD = trainD[permutation]
            trainH = trainH[permutation]
            trainM = trainM[permutation]
            trainL = trainL[permutation]
            trainXAll = trainXAll[permutation]

            for batch_idx in range(num_batch):
                iteration += 1
                start_idx = batch_idx * self.batch_size
                end_idx = min(shape[0], (batch_idx + 1) * self.batch_size)

                xs = np.expand_dims(trainX[start_idx:end_idx], axis=-1)
                d_of_week = np.reshape(trainDoW[start_idx:end_idx], [-1, self.site_num])
                day = np.reshape(trainD[start_idx:end_idx], [-1, self.site_num])
                hour = np.reshape(trainH[start_idx:end_idx], [-1, self.site_num])
                minute = np.reshape(trainM[start_idx:end_idx], [-1, self.site_num])
                labels = trainL[start_idx:end_idx]
                xs_all = np.expand_dims(trainXAll[start_idx:end_idx], axis=-1)

                # Convert to tensors
                xs = tf.constant(xs, dtype=tf.float32)
                xs_all = tf.constant(xs_all, dtype=tf.float32)
                d_of_week = tf.constant(d_of_week, dtype=tf.int32)
                day = tf.constant(day, dtype=tf.int32)
                hour = tf.constant(hour, dtype=tf.int32)
                minute = tf.constant(minute, dtype=tf.int32)
                labels = tf.constant(labels, dtype=tf.float32)

                l1, l2 = self.train_step(xs, xs_all, d_of_week, day, hour, minute, labels)

                if iteration % 100 == 0:
                    end_time = datetime.datetime.now()
                    total_time = end_time - start_time
                    print("Total running times is : %f" % total_time.total_seconds())

            print('validation')
            mae = self.evaluate(valX, valDoW, valD, valH, valM, valL, valXAll)
            if max_mae > mae:
                print("in the %dth epoch, the validate average loss value is : %.3f" % (epoch + 1, mae))
                max_mae = mae
                # Save checkpoint
                checkpoint = tf.train.Checkpoint(model=self)
                checkpoint.save(file_prefix=self.para.save_path)

    def evaluate(self, testX, testDoW, testD, testH, testM, testL, testXAll):
        '''
        :return:
        '''
        labels_list, pres_list = list(), list()

        if not self.is_training:
            checkpoint = tf.train.Checkpoint(model=self)
            latest = tf.train.latest_checkpoint(os.path.dirname(self.para.save_path))
            if latest:
                print('the model weights has been loaded:')
                checkpoint.restore(latest)

        parameters = sum(np.prod(v.shape) for v in self.trainable_variables)
        print('trainable parameters: {:,}'.format(parameters))

        textX_shape = testX.shape
        total_batch = math.ceil(textX_shape[0] / self.batch_size)
        start_time = datetime.datetime.now()
        for b_idx in range(total_batch):
            start_idx = b_idx * self.batch_size
            end_idx = min(textX_shape[0], (b_idx + 1) * self.batch_size)

            xs = np.expand_dims(testX[start_idx:end_idx], axis=-1)
            d_of_week = np.reshape(testDoW[start_idx:end_idx], [-1, self.site_num])
            day = np.reshape(testD[start_idx:end_idx], [-1, self.site_num])
            hour = np.reshape(testH[start_idx:end_idx], [-1, self.site_num])
            minute = np.reshape(testM[start_idx:end_idx], [-1, self.site_num])
            labels = testL[start_idx:end_idx]
            xs_all = np.expand_dims(testXAll[start_idx:end_idx], axis=-1)

            xs = tf.constant(xs, dtype=tf.float32)
            xs_all = tf.constant(xs_all, dtype=tf.float32)
            d_of_week = tf.constant(d_of_week, dtype=tf.int32)
            day = tf.constant(day, dtype=tf.int32)
            hour = tf.constant(hour, dtype=tf.int32)
            minute = tf.constant(minute, dtype=tf.int32)

            pre_s = self.eval_step(xs, xs_all, d_of_week, day, hour, minute)

            labels_list.append(labels[:, :, self.input_len:])
            pres_list.append(pre_s.numpy()[:, :, self.input_len:])

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

        labels_list = np.concatenate(labels_list, axis=0)
        pres_list = np.concatenate(pres_list, axis=0)

        print('                MAE\t\tRMSE\t\tMAPE')
        if not self.is_training:
            for i in range(self.para.output_length):
                mae, rmse, mape = metric(pres_list[:, :, i], labels_list[:, :, i])
                print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, mae, rmse, mape * 100))
        mae, rmse, mape = metric(pres_list, labels_list)
        print('average:         %.3f\t\t%.3f\t\t%.3f%%' % (mae, rmse, mape * 100))

        return mae


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

    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll, mean, std = loadData(para)
    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainL.shape))
    print('valX:   %s\t\tvalY:   %s' % (valX.shape, valL.shape))
    print('testX:  %s\t\ttestY:  %s' % (testX.shape, testL.shape))
    print('data loaded!')
    pre_model = Model(para, mean, std)

    if int(val) == 1:
        pre_model.run_epoch(trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll,
                           valX, valDoW, valD, valH, valM, valL, valXAll,
                           testX, testDoW, testD, testH, testM, testL, testXAll)
    else:
        pre_model.evaluate(testX, testDoW, testD, testH, testM, testL, testXAll)

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()