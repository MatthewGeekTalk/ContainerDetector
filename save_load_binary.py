import tensorflow as tf
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
materials = os.listdir(os.path.abspath('./Negative train data'))

sess = tf.Session()
dir(tf.contrib)
saver = tf.train.import_meta_graph('./module/bc-cnn/500/binary_classification_CNN.ckpt.meta')
saver.restore(sess, './module/bc-cnn/500/binary_classification_CNN.ckpt')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("output/predict_sm:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

if __name__ == '__main__':
    for i in range(len(materials)):
        path = os.path.abspath('./Negative train data') + os.path.sep + str(materials[i])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.reshape(img,[1,2352])
        feed_dict = {x: img, keep_prob: 1.0}
        print(sess.run(y, feed_dict=feed_dict))



