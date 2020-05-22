import tensorflow as tf
from PIL import Image
import numpy as np
from tqdm import tqdm
import os


tf.compat.v1.flags.DEFINE_string('image_dir', '', 'Path to celeb_a images')
tf.compat.v1.flags.DEFINE_string('out_dir', '', 'Path where tfrecord files are stored')
tf.compat.v1.flags.DEFINE_bool('precropped', False, 'Whether images are pre-cropped')
FLAGS = tf.compat.v1.flags.FLAGS

write_options = tf.io.TFRecordOptions(compression_type='GZIP',
                                      compression_level=9)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tfrecords():
    with tf.compat.v1.python_io.TFRecordWriter(os.path.join(FLAGS.out_dir, 'celeb_a.tfrecords'),
                                               options=write_options) as writer:
        for img in tqdm(os.listdir(FLAGS.image_dir)):
            image_path = os.path.join(FLAGS.image_dir, img)
            crop = (30, 55, 150, 175)
            image_array = np.array((Image.open(image_path).crop(crop)).resize((64, 64))) if not FLAGS.precropped \
                else np.array(Image.open(image_path))
            height = image_array.shape[0]
            width = image_array.shape[1]
            channels = image_array.shape[2]

            raw_image = image_array.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'image_raw': _bytes_feature(raw_image)}))

            writer.write(example.SerializeToString())

    writer.close()


def main(_):
    make_tfrecords()


if __name__ == '__main__':
    tf.compat.v1.app.run()
