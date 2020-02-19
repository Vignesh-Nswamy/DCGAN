import tensorflow as tf


class Discriminator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = tf.keras.Sequential()
        self.weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02, mean=0.0, seed=42)

    def add_conv_stack(self, out_channels: int, filter_size: tuple, strides: tuple, padding='same'):
        self.model.add(tf.keras.layers.Conv2D(out_channels, filter_size, strides=strides, padding=padding,
                                              kernel_initializer=self.weight_initializer))
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.3))

    def get_model(self):
        self.model.add(tf.keras.layers.Input(self.input_shape))

        self.add_conv_stack(64, (5, 5), (2, 2))

        self.add_conv_stack(128, (5, 5), (2, 2))

        self.add_conv_stack(256, (5, 5), (2, 2))

        self.add_conv_stack(512, (5, 5), (2, 2))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1, kernel_initializer=self.weight_initializer))

        return self.model

    def get_loss(self, real_output, fake_output):
        real_loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output,
                                                                                labels=tf.ones_like(real_output)))
        fake_loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                                labels=tf.zeros_like(fake_output)))
        return real_loss + fake_loss


if __name__ == '__main__':
    discriminator = Discriminator((64, 64, 3))
    model = discriminator.get_model()
    print(model.summary())
