from __future__ import print_function
import os
import sys
os.environ['KERAS_BACKEND'] = "tensorflow"
################################################################################################
################################################################################################
################################################################################################
import numpy as np
import tensorflow as tf
import random as rn

os.environ['PYTHONHASHSEED'] = '0'

from keras import backend as K
K.set_image_data_format('channels_first')

np.random.seed(27)
rn.seed(27)
tf.set_random_seed(27)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
################################################################################################
################################################################################################
################################################################################################
import keras
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
from bezier_surface_fitting import bsfit

img_rows = 32
img_cols = 32
img_channels = 3
col_scale_div = 255

(x_test, y_test), (x_train, y_train) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= col_scale_div
x_test /= col_scale_div

n_img = 16
img_batch = x_train[:n_img,:]
m = 40
n = 40

model = bsfit(img_batch, m, n)
dummy_input = np.zeros((1, 1)).astype(np.float32)

pred = model.predict(dummy_input)


# for analysis
mydir = "analysis/analysis_"+str(m)+"_"+str(n)
os.makedirs(mydir)
for i in range(n_img):
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(np.transpose(img_batch[i,:], [1,2,0]), cmap="gray")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(np.transpose(np.clip(pred[0, i,:],0,1), [1,2,0]), cmap="gray")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(np.transpose(np.abs(img_batch[i,:]-np.clip(pred[0, i,:],0,1)), [1,2,0]), cmap="gray")
    plt.savefig(mydir+"/bsf" + str(i) + "_" + str(m) + "_" + str(n) + ".png")
    plt.close()

