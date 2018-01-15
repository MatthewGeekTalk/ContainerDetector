import tensorflow as tf
import tensorflow.contrib.slim as slim

'''keep inputs as (336, 448, 3)'''


def ocr_net(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        # 336 * 448 * 3
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [6, 6], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # 168 * 224 * 64
        net = slim.repeat(net, 2, slim.conv2d, 128, [6, 6], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # 84 * 112 * 128
        net = slim.repeat(net, 3, slim.conv2d, 256, [6, 6], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # 42 * 56 * 256
        net = slim.repeat(net, 3, slim.conv2d, 512, [6, 6], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # 21 * 28 * 512
        net = slim.repeat(net, 3, slim.conv2d, 512, [6, 6], scope='conv5')
        # 21 * 28 * 512
        net = slim.fully_connected(net, 32768, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 16348, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 4096, scope='fc8')
        net = slim.dropout(net, 0.5, scope='dropout8')
        net = slim.fully_connected(net, 2, activation_fn=None, scope='fc8')
    return net


class ocr_handler(object):
    def __init__(self, learning_rate, epochs, sum_secs, inter_secs):
        self.LR = learning_rate
        self.epochs = epochs
        self.sum_secs = sum_secs
        self.inter_secs = inter_secs

    @staticmethod
    def _build_net(inputs):
        return ocr_net(inputs=inputs)

    def _train_model(self, predicts, labels, logdir):
        loss = slim.losses.softmax_cross_entropy(predicts, labels)
        optimizer = tf.train.GradientDescentOptimizer(self.LR)
        train_op = slim.learning.create_train_op(loss, optimizer)

        slim.learning.train(
            train_op,
            logdir,
            number_of_steps=self.epochs,
            save_summaries_secs=self.sum_secs,
            save_interval_secs=self.inter_secs)

    def main(self, imgs, labels, logdir):
        predictions = ocr_net(imgs)
        self._train_model(predictions, labels, logdir)
