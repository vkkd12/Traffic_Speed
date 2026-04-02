# -- coding: utf-8 --
from models.spatial_attention import SpatialTransformer
from models.inits import *
from models.temporal_attention import TemporalTransformer
from baseline.gman import tf_utils
from models.utils import *

class ST_Block(tf.Module):
    def __init__(self, hp=None, placeholders=None, input_length=6, model_func=None, supports=None):
        super().__init__(name='st_block')
        self.para = hp
        self.batch_size = self.para.batch_size
        self.emb_size = self.para.emb_size
        self.site_num = self.para.site_num
        self.is_training = self.para.is_training
        self.dropout = self.para.dropout
        self.features = self.para.features
        self.placeholders = placeholders
        self.input_length = input_length
        self.model_func = model_func
        self.supports = supports

        # Pre-create internal blocks to avoid recreating variables in @tf.function traces
        self.S = SpatialTransformer(self.para)
        self.T = TemporalTransformer(self.para)
        
        if self.para.model_name in ['MT-STGIN', 'MT-STGIN-3', 'MT-STGIN-4']:
            self.gcn = self.model_func(self.placeholders,
                                    input_dim=self.emb_size * 1,
                                    para=self.para,
                                    supports=self.supports)
        else:
            self.gcn = None

    def spatio_temporal_(self, speed=None, STE=None, supports=None):
        # STAttBlock is unsupported and deprecated in TF2 migration for now.
        return speed

    def spatio_temporal(self, speed=None, STE=None, supports=None, causality=False):
        '''
        :param features: [N, site_num, emb_size]
        :param day: [N, input_length, site_num, emb_size]
        :return: [N, input_length, site_num, emb_size]
        '''
        x = tf.concat([speed, STE], axis=-1)
        """
        dynamic spatial correlation
        """
        x_s = tf.reshape(x, shape=[-1, self.site_num, self.emb_size * 2])
        x_s = self.S.encoder(inputs=x_s)
        x_f = tf.reshape(x_s, shape=[-1, self.para.input_length, self.site_num, self.emb_size])

        """
        physical relationship extraction with GCN
        """
        if self.gcn is not None:
            # We must pass supports explicitly during forward if GCN uses it
            # But the existing model_func instantiation bound it. In TF2, GCN should take supports in call()
            # Assuming predict() can accept a dense tensor.
            x_g = tf.reshape(speed, shape=[-1, self.site_num, self.emb_size * 1])
            x_g = self.gcn.predict(x_g)
            x_g = tf.reshape(x_g, shape=[-1, self.para.input_length, self.site_num, self.emb_size])

        """
        spatial - fusion gating mechanism
        """
        if self.para.model_name != 'MT-STGIN-2':
            if self.para.model_name != 'MT-STGIN-3':
                z = tf.nn.sigmoid(tf.multiply(x_f, x_g))
                x_f = tf.add(tf.multiply(z, x_f), tf.multiply(1 - z, x_g))
            else:
                x_f = tf.add(x_f, x_g)

        """
        dynamic temporal correlation extraction with attention mechanism
        """
        x_t = tf.transpose(speed, perm=[0, 2, 1, 3])
        x_t = tf.reshape(x_t, shape=[-1, self.input_length, self.emb_size * 1])
        x_t = self.T.encoder(hiddens=x_t, hidden=x_t, causality=causality)

        x_t = tf.reshape(x_t, shape=[-1, self.site_num, self.input_length, self.emb_size])
        x_t = tf.transpose(x_t, perm=[0, 2, 1, 3])

        """
        spatial and temporal - fusion gating mechanism
        """
        if self.para.model_name != 'MT-STGIN-3':
            z = tf.nn.sigmoid(tf.multiply(x_f, x_t))
            x_f = tf.add(tf.multiply(z, x_f), tf.multiply(1 - z, x_t))
        else:
            x_f = tf.add(x_f, x_t)
        return x_f  # [N, input_length, site_num, emb_size]


    def dynamic_spatio_temporal(self, speed=None, STE=None, supports=None, causality=False):
        '''
        :return: [N, 1, site_num, emb_size]
        '''
        x = tf.concat([speed, STE], axis=-1)
        """
        dynamic spatial correlation
        """
        x_s = tf.reshape(x, shape=[-1, self.site_num, self.emb_size * 2])
        x_s = self.S.encoder(inputs=x_s)
        x_f = tf.reshape(x_s, shape=[-1, 1, self.site_num, self.emb_size])

        """
        physical relationship extraction with GCN
        """
        if self.gcn is not None:
            x_g = tf.reshape(speed, shape=[-1, self.site_num, self.emb_size * 1])
            x_g = self.gcn.predict(x_g)
            x_g = tf.reshape(x_g, shape=[-1, 1, self.site_num, self.emb_size])

            """
            spatial - fusion gating mechanism
            """
            z = tf.nn.sigmoid(tf.multiply(x_f, x_g))
            x_f = tf.add(tf.multiply(z, x_f), tf.multiply(1 - z, x_g))

        """
        dynamic temporal correlation
        """
        x_t = tf.transpose(speed, perm=[0, 2, 1, 3])
        x_t = tf.reshape(x_t, shape=[-1, 1, self.emb_size * 1])
        x_t = self.T.encoder(hiddens=x_t, hidden=x_t, causality=causality)

        x_t = tf.reshape(x_t, shape=[-1, self.site_num, 1, self.emb_size])
        x_t = tf.transpose(x_t, perm=[0, 2, 1, 3])

        """
        spatial and temporal - fusion gating mechanism
        """
        z = tf.nn.sigmoid(tf.multiply(x_f, x_t))
        x_f = tf.add(tf.multiply(z, x_f), tf.multiply(1 - z, x_t))
        return x_f  # [N, 1, site_num, emb_size]