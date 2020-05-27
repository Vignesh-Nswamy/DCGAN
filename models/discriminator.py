import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential


class Discriminator:
    def __init__(self, configs):
        self.configs = configs
        self.weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02, mean=0.0, seed=42)

        self.input_dim = eval(self.configs.img_dim)

    def __add_conv_block(self, model, filters, kernel_size, strides, block_num):
        model.add(Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         activation=LeakyReLU(),
                         kernel_initializer=self.weight_initializer,
                         name=f'conv_{block_num}'))
        model.add(Dropout(0.3,
                          name=f'dropout_{block_num}'))

    def __create(self):
        model = Sequential()
        model.add(Input(self.input_dim,
                        name='in_layer'))
        for block_num, num_filters in enumerate([64, 128, 256, 512]):
            self.__add_conv_block(model,
                                  num_filters,
                                  5,
                                  2,
                                  block_num)

        model.add(Flatten(name='flatten'))
        model.add(Dense(1,
                        kernel_initializer=self.weight_initializer,
                        name='out_layer'))
        return model

    def model(self):
        ckpt_dir = self.configs.checkpoints.path
        if os.path.exists(os.path.join(ckpt_dir, 'discriminator.ckpt')):
            print(f'Found saved discriminator at {os.path.join(ckpt_dir, "discriminator.ckpt")}. Loading...')
            return load_model(os.path.join(ckpt_dir, "discriminator.ckpt"))
        else:
            return self.__create()
