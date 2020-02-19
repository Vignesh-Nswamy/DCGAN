import tensorflow as tf


class Generator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = tf.keras.Sequential()
        self.weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02, mean=0.0, seed=42)

    def add_convT_stack(self, out_channels: int, filter_size: tuple, strides: tuple, padding='same', use_bias=False):
        self.model.add(tf.keras.layers.Conv2DTranspose(out_channels, filter_size, strides=strides, padding=padding,
                                                       use_bias=use_bias, kernel_initializer=self.weight_initializer))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

    def get_model(self):
        self.model.add(tf.keras.layers.Dense(4 * 4 * 1024, input_shape=self.input_shape,
                                             kernel_initializer=self.weight_initializer))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Reshape((4, 4, 1024)))

        self.add_convT_stack(512, (5, 5), (2, 2))

        self.add_convT_stack(256, (5, 5), (2, 2))

        self.add_convT_stack(128, (5, 5), (2, 2))

        # self.add_convT_stack(64, (5, 5), (2, 2))

        self.model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                       kernel_initializer=self.weight_initializer, activation='tanh'))
        return self.model

    def get_loss(self, fake_output):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                      labels=tf.ones_like(fake_output)))


if __name__ == '__main__':
    generator = Generator((100,))
    model = generator.get_model()
    print(model.summary())
