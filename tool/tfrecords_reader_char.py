import tensorflow as tf
import os
import tensorflow.contrib.data as tcd
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class tfrecords_reader_char:
    def __init__(self, path):
        self.tfrecord_path = path

    def _parse_function(self, example_proto):
        keys_to_features = {'train/image': tf.FixedLenFeature([], tf.string),
                            'train/label': tf.FixedLenFeature([35], dtype=tf.int64)}
        features = tf.parse_single_example(example_proto, features=keys_to_features)
        images = tf.decode_raw(features['train/image'], tf.uint8)
        labels = tf.cast(features['train/label'], tf.int32)
        images = tf.reshape(images, [28, 28])
        return images, labels

    def main(self, batch):
        # data_path = self.tfrecord_path + os.path.sep + 'chars.tfrecords'
        data_path = self.tfrecord_path
        dataset = tcd.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_function)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images, labels = sess.run([image_batch, label_batch])
            coord.request_stop()
            coord.join(threads)

        return images, labels

if __name__ == '__main__':
    path = os.path.abspath('../TFRecords')
    reader = tfrecords_reader_char(path)
    imgs, labels = reader.main(50)
    print(imgs.shape, labels.shape)
    plt.imshow(imgs[2])
    plt.show()
