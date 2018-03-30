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
from bezier_surface_fitting import bs_fit, bez_filter, bez_filter_log
from bezier_surface_plotter import bsplot

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

n_img = 4
img_batch = x_train[:n_img,:]
m = 32
n = 32
n_epochs = 100

bez_input = bez_filter(img_rows, img_cols, m, n)
models = []
for i in range(n_img):
    models.append(bs_fit(img_batch[i,:], bez_input, m, n, epochs=n_epochs))


preds = []
for i in range(n_img):
    preds.append(np.reshape(models[i].predict(bez_input), (img_rows,img_cols,img_channels)))


# for analysis
H = img_rows
W = img_cols
fine_bez_input = bez_filter(H, W, m, n)
fine_preds = []
for i in range(n_img):
    fine_preds.append(np.reshape(models[i].predict(fine_bez_input), (H,W,img_channels)))


log_H = 100
log_W = 100
log_h_end = (img_rows-0.5)/img_rows
log_w_end = (img_cols-0.5)/img_cols
log_bez_input = bez_filter_log(log_H, log_W, m, n, log_h_end, log_w_end)
log_preds = []
for i in range(n_img):
    log_preds.append(np.reshape(models[i].predict(log_bez_input), (log_H,log_W,img_channels)))


mydir = "analysis/log_2_big_img_sep_models_analysis_"+str(m)+"_"+str(n)
if not os.path.exists(mydir):
    os.makedirs(mydir)


for i in range(n_img):
    # bsplot(m, n, H, W, fine_preds[i], mydir+"/bsf_full_" + str(i) + "_" + str(m) + "_" + str(n) + ".png")
    fig=plt.figure(figsize=(20, 10))
    columns = 2
    rows = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(np.transpose(img_batch[i,:], [1,2,0]), cmap="gray")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(np.clip(preds[i],0,1), cmap="gray")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(np.clip(log_preds[i],0,1), cmap="gray")
    fig.add_subplot(rows, columns, 4)
    plt.imshow(np.abs(np.transpose(img_batch[i,:], [1,2,0])-np.clip(preds[i],0,1)), cmap="gray")
    plt.savefig(mydir+"/bsf_" + str(i) + "_" + str(m) + "_" + str(n) + ".png")
    plt.close()