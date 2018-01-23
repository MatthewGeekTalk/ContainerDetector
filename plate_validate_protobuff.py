import tensorflow as tf
import os
import cv2
import numpy as np
from Graph import Graph

FREEZE_MODEL_PATH = os.path.abspath('./frozen_module/bc-cnn2')


class PlateValidate(object):
    def __init__(self):
        dir(tf.contrib)
        self.imgs = []
        self.in_imgs = []
        self.imgs_labels = []

    @staticmethod
    def __get_imgs(imgs,batch):
        imgs_list = []
        if batch == 0:
            img = cv2.resize(imgs, (28, 28), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.reshape(img, [-1, 2352])
            imgs_list.append(img)
        elif batch == 1:
            for img in imgs:
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.reshape(img, [-1, 2352])
                imgs_list.append(img)
        return imgs_list

    @staticmethod
    def __load_graph(frozen_graph):
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')
        return graph

    def __plate_validate(self):
        graph = Graph()
        sess = graph.sess_bc
        x = sess.graph.get_tensor_by_name('prefix/x:0')
        y = sess.graph.get_tensor_by_name('prefix/output/predict_sm:0')
        keep_prob = sess.graph.get_tensor_by_name('prefix/keep_prob:0')

        for i in range(len(self.in_imgs)):
            logits = sess.run(y, feed_dict={
                x: self.in_imgs[i], keep_prob: 1.0
            })

            logits = np.reshape(logits, [2])
            logits = np.asarray(logits, dtype=np.int32)
            self.imgs_labels.append(list(logits))

    def main(self, imgs, batch):
        self.imgs_labels = []
        self.imgs = imgs
        self.in_imgs = self.__get_imgs(imgs=imgs,batch=batch)
        self.__plate_validate()
        return self.imgs, self.imgs_labels


if __name__ == '__main__':
    img_list = []
    path = input('Please input your image path:')

    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    img_list.append(img)

    plate_validate = PlateValidate()
    imgs, labels = plate_validate.main(img_list)

    print(labels)
