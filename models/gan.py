from .discriminator import Discriminator
from .generator import Generator

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

from utils.util import get_data
import matplotlib.pyplot as plt

import os
from tensorflow.keras.utils import Progbar



class Gan:
    def __init__(self, configs):
        self.configs = configs

        self.__lr = eval(self.configs.learning.rate)
        self.__save_model = self.configs.checkpoints.save
        self.__save_interval = self.configs.checkpoints.save_interval

        self.discriminator = Discriminator(self.configs).model()
        self.generator = Generator(self.configs).model()

        self.generator_opt, self.discriminator_opt = (RMSprop(self.__lr), RMSprop(self.__lr)) \
            if self.configs.learning.optimizer == 'rmsprop'\
            else (Adam(self.__lr, beta_1=0.5), Adam(self.__lr, beta_1=0.5))

        self.num_epochs = self.configs.num_epochs

        self.__create_dirs()

    def __create_dirs(self):
        if not os.path.exists(self.configs.checkpoints.path):
            print(f'{self.configs.checkpoints.path} not found. Creating....')
            os.makedirs(self.configs.checkpoints.path, exist_ok=True)
        if not os.path.exists(self.configs.results.path):
            print(f'{self.configs.results.path} not found. Creating....')
            os.makedirs(self.configs.results.path, exist_ok=True)
            os.makedirs(os.path.join(self.configs.results.path, 'images'), exist_ok=True)
        if not os.path.exists(os.path.join(self.configs.results.path, 'images')):
            os.makedirs(os.path.join(self.configs.results.path, 'images'), exist_ok=True)

    @staticmethod
    def get_discriminator_loss(real_output, fake_output):
        real_loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output,
                                                                                labels=tf.ones_like(real_output)))
        fake_loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                                labels=tf.zeros_like(fake_output)))
        return real_loss + fake_loss

    @staticmethod
    def get_generator_loss(fake_output):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                      labels=tf.ones_like(fake_output)))

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

    def train(self):
        @tf.function
        def train_step(image_batch):
            noise = tf.random.normal((self.configs.data.batch_size, eval(self.configs.noise_dim)[0]))
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(image_batch, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                generator_loss = self.get_generator_loss(fake_output)
                discriminator_loss = self.get_discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
            self.generator_opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            self.discriminator_opt.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            return [('generator_loss', generator_loss.numpy()), ('discriminator_loss', discriminator_loss.numpy())]
        data = get_data(self.configs.data.path, self.configs.data.batch_size)
        for i in range(self.num_epochs):
            print("\nepoch {}/{}".format(i + 1, self.num_epochs))
            prog_bar = Progbar(None, stateful_metrics=['generator_loss', 'discriminator_loss'])
            for idx, im_batch in enumerate(data):
                losses = train_step(im_batch)
                prog_bar.update(idx+1, values=losses)
            self.save_results(i)
            if self.__save_model and (i+1) % self.__save_interval == 0:
                self.save_models()
