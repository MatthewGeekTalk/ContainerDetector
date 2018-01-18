import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('111.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.reshape(img,[1,2352])
    sess = tf.Session()
    dir(tf.contrib)
    saver = tf.train.import_meta_graph('binary_classification_CNN.ckpt.meta')
    saver.restore(sess, './binary_classification_CNN.ckpt')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("output/predict_sm:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    feed_dict = {x: img, keep_prob: 1.0}
    print(sess.run(y, feed_dict=feed_dict))



