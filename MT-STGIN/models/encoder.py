import tensorflow as tf
from models.st_block import ST_Block


class Encoder_ST(tf.Module):
    def __init__(self, hp, placeholders=None, model_func=None, supports=None):
        '''
        :param hp:
        '''
        super().__init__(name='encoder_st')
        self.para = hp
        self.input_length = self.para.input_length
        self.placeholders = placeholders
        self.model_func = model_func
        self.supports = supports
        self.st_block = ST_Block(hp=self.para, placeholders=self.placeholders, input_length=self.input_length, model_func=self.model_func, supports=self.supports)

    def encoder_spatio_temporal(self, speed=None, STE=None):
        '''
        :param features: [N, site_num, emb_size]
        :param STE: [N, input_length, site_num, emb_size]
        :return: [N, input_length, site_num, emb_size]
        '''
        # dynamic spatial correlation

        result = self.st_block.spatio_temporal(speed=speed,
                                              STE=STE,
                                              supports=self.supports)

        return result  # [N, input_length, site_num, emb_size]