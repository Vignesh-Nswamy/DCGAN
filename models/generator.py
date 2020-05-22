import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential


class Generator:
    def __init__(self, configs):
        self.configs = configs
        self.__kernel_regularizer = tf.keras.regularizers.l2(0.001)
        self.__he_normal_initializer = tf.keras.initializers.he_normal(seed=0)
        self.__lecun_normal_initializer = tf.keras.initializers.lecun_normal(seed=0)

        self.input_dim = eval(self.configs.noise_dim)

    def __add_conv_block(self, model, filters, kernel_size, strides, block_num):
        model.add(Conv2DTranspose(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  activation=LeakyReLU(),
                                  use_bias=False,
                                  kernel_regularizer=self.__kernel_regularizer,
                                  kernel_initializer=self.__he_normal_initializer,
                                  name=f'conv_T_{block_num}'))
        model.add(BatchNormalization(name=f'batch_norm_{block_num}'))

    def __create(self):
        model = Sequential()
        model.add(Dense(4 * 4 * 1024,
                        input_shape=self.input_dim,
                        activation=LeakyReLU(),
                        kernel_regularizer=self.__kernel_regularizer,
                        kernel_initializer=self.__he_normal_initializer,
                        name='fc_1'))
        model.add(BatchNormalization(name='batch_norm'))
        model.add(Reshape((4, 4, 1024),
                          name='reshape'))

        for block_num, num_filters in enumerate([512, 256, 128]):
            self.__add_conv_block(model, num_filters, 5, 2, block_num)

        model.add(Conv2DTranspose(filters=3,
                                  kernel_size=5,
                                  strides=2,
                                  padding='same',
                                  activation='tanh',
                                  use_bias=False,
                                  kernel_regularizer=self.__kernel_regularizer,
                                  kernel_initializer=self.__lecun_normal_initializer))

        return model

    def model(self):
        ckpt_dir = self.configs.checkpoints.path
        if os.path.exists(os.path.join(ckpt_dir, 'generator.ckpt')):
            print(f'Found saved generator at {os.path.join(ckpt_dir, "generator.ckpt")}. Loading...')
            return load_model(os.path.join(ckpt_dir, "generator.ckpt"))
        else:
            return self.__create()

