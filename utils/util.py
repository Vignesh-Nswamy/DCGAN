import tensorflow as tf

from .luatables import LuaEmulator


def get_data(data_path, batch_size):
    def decode(serialized_example):
        features = tf.compat.v1.parse_single_example(
            serialized_example,
            features={
                'height': tf.compat.v1.FixedLenFeature([], tf.int64),
                'width': tf.compat.v1.FixedLenFeature([], tf.int64),
                'channels': tf.compat.v1.FixedLenFeature([], tf.int64),
                'image_raw': tf.compat.v1.FixedLenFeature([], tf.string)
            })
        # height = tf.cast(features['height'], tf.int32)
        # width = tf.cast(features['width'], tf.int32)
        # channels = tf.cast(features['channels'], tf.int32)
        image = tf.compat.v1.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, (64, 64, 3))
        return image

    def normalize(image):
        image = tf.cast(image, tf.float32) / 255.
        return image

    return tf.data.TFRecordDataset(data_path, compression_type='GZIP').map(decode).map(normalize).batch(batch_size)


def convert(yaml_dict):
    """Converts a dictionary to a LuaTable-like object."""
    if isinstance(yaml_dict, dict):
        yaml_dict = LuaEmulator(yaml_dict)
    for key, item in yaml_dict.items():
        if isinstance(item, dict):
            yaml_dict[key] = convert(item)
    return yaml_dict


def merge(config: dict,
          flags):
    for key in flags:
        config[key] = flags[key].value
    return convert(config)
