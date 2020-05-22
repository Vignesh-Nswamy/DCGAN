from .discriminator import Discriminator
from .generator import Generator

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

from utils.util import get_data
import matplotlib.pyplot as plt

import os


class Gan:
    def __init__(self, configs):
        self.configs = configs

        self.__lr = eval(self.configs.learning.rate)
        self.__save_model = eval(self.configs.checkpoints.save)
        self.__save_interval = eval(self.configs.checkpoints.save_interval)

        self.discriminator = Discriminator(self.configs).model()
        self.generator = Generator(self.configs).model()

        self.generator_opt, self.discriminator_opt = RMSprop(self.__lr), RMSprop(self.__lr) \
            if self.configs.learning.optimizer == 'rmsprop'\
            else Adam(self.__lr, beta_1=0.5), Adam(self.__lr, beta_1=0.5)

        self.num_epochs = eval(self.configs.num_epochs)

        self.__create_dirs()

    def __create_dirs(self):
        if not os.path.exists(self.configs.checkpoints.path):
            print(f'{self.configs.checkpoints.path} not found. Creating....')
            os.makedirs(self.configs.checkpoints.path, exist_ok=True)
        if not os.path.exists(self.configs.results.path):
            print(f'{self.configs.results.path} not found. Creating....')
            os.makedirs(self.configs.results.path, exist_ok=True)
            os.makedirs(os.path.join(self.configs.results.path, 'images'), exist_ok=True)

    @staticmethod
    def __get_losses(fake_output, real_output):
        # Generator loss
        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                                labels=tf.ones_like(fake_output)))

        # Discriminator loss
        real_loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output,
                                                                                labels=tf.ones_like(real_output)))
        fake_loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                                labels=tf.zeros_like(fake_output)))
        discriminator_loss = real_loss + fake_loss

        return generator_loss, discriminator_loss

    def save_models(self):
        self.generator.save(os.path.join(self.configs.checkpoints.path, 'generator.ckpt'))
        self.discriminator.save(os.path.join(self.configs.checkpoints.path, 'discriminator.ckpt'))

    def save_results(self, epoch):
        path = self.configs.results.path
        noise = tf.random.normal([25, 100], seed=0)
        generated_imgs = self.generator(noise, training=False).numpy()
        plt.figure(figsize=(18, 18))
        for i in range(generated_imgs.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.imshow(generated_imgs[i, :, :, :])
            plt.axis('off')
        plt.savefig(os.path.join(path, 'images', 'epoch_{:04d}.png'.format(epoch)))
        plt.close()

    @tf.function
    def __train_step(self, image_batch):
        noise = tf.random.normal()
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(image_batch, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            generator_loss, discriminator_loss = self.__get_losses(fake_output, real_output)

        gradients_of_generator = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self):
        for i in range(self.num_epochs):
            for image_batch in get_data(self.configs.data.path, eval(self.configs.data.batch_size)):
                self.train(image_batch)
                self.save_results(i)
            if self.__save_model and (i+1) % self.__save_interval == 0:
                self.save_models()
