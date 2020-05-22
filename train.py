import tensorflow as tf
from models import Gan
from utils import util
import yaml


tf.compat.v1.flags.DEFINE_string('config_path', '', 'Path to a YAML configuration files defining FLAG values.')
tf.compat.v1.flags.DEFINE_integer('num_epochs', 75, 'Number of training epochs')
FLAGS = tf.compat.v1.flags.FLAGS


def main(_):
    config = yaml.load(open(FLAGS.config_path), Loader=yaml.FullLoader)
    config = util.merge(config,
                        FLAGS)
    gan = Gan(config)

    gan.train()


if __name__ == '__main__':
    tf.compat.v1.app.run()
