import tensorflow as tf
from models.st_block import ST_Block
from models.bridge import BridgeTransformer

class Decoder_ST(tf.Module):
    def __init__(self, hp, placeholders=None, model_func=None, supports=None):
        '''
        :param hp:
        '''
        super().__init__(name='decoder_st')
        self.para = hp
        self.output_length = self.para.output_length
        self.placeholders = placeholders
        self.model_func = model_func
        self.supports = supports

        if self.para.model_name != 'MT-STGIN-4':
            self.st_block = ST_Block(hp=self.para, placeholders=self.placeholders, input_length=self.output_length + self.para.pre_length,
                            model_func=self.model_func, supports=self.supports)
        else:
            self.bridge_encoder = BridgeTransformer(self.para)
            self.st_block = ST_Block(hp=self.para, placeholders=self.placeholders, input_length=self.output_length,
                            model_func=self.model_func, supports=self.supports)


    def decoder_spatio_temporal(self, speed=None, STE=None, causality=False):
        '''
        :param speed: [N, time length, site_num, emb_size]
        :return: [N, output_length, site_num, emb_size]
        '''
        # dynamic spatial correlation
        if self.para.model_name != 'MT-STGIN-4':
            result = self.st_block.spatio_temporal(speed=speed,
                                              STE=STE,
                                              supports=self.supports, causality=causality)
        else:
            result = list()
            for time_step in range(self.output_length):
                with tf.name_scope('dynamic_decoding_bridge'):
                    bridge_outs = self.bridge_encoder.encoder(X=speed,
                                                         X_P=speed,
                                                         X_Q=STE[:, time_step:time_step+1],
                                                         causality=False)

                with tf.name_scope('dynamic_decoding_spatio_temporal'):
                    each_time_step_hidden = self.st_block.dynamic_spatio_temporal(speed=bridge_outs,
                                                                             STE=STE[:, time_step:time_step+1],
                                                                             supports=self.supports, causality=causality)
                speed = tf.concat([speed, each_time_step_hidden], axis=1)
                result.append(each_time_step_hidden)
            result = tf.concat(result, axis=1)

        return result  # [N, output_length, site_num, emb_size]