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

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session()
K.set_session(sess)
################################################################################################
################################################################################################
################################################################################################
import keras
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
from bezier_surface_fitting import bs_fit, bez_filter, bs_build_model, bs_get_weights, bs_set_weights
from bezier_surface_plotter import bsplot
import cv2

fname = "data/tree.png"
img = cv2.imread(fname)
img = cv2.resize(img, (int(img.shape[0]/4),int(img.shape[1]/4)))

img_rows = img.shape[0]
img_cols = img.shape[1]
img_channels = img.shape[2]
col_scale_div = 255

img = np.transpose(img, [2,0,1])
img = img.reshape((1,img_channels,img_rows,img_cols))
img = img.astype('float32')
img /= col_scale_div

n_img = 1
img_batch = img
m = 40
n = 40

bez_input = bez_filter(img_rows, img_cols, m, n)
models = []
for i in range(n_img):
    models.append(bs_fit(img_batch[i,:], bez_input, m, n))


preds = []
for i in range(n_img):
    preds.append(np.reshape(models[i].predict(bez_input), (img_rows,img_cols,img_channels)))


# for analysis
bez_input = None    # free up memory
H = 2*img_rows
W = 2*img_cols
fine_bez_input = bez_filter(H, W, m, n)
fine_preds = []
for i in range(n_img):
    fine_preds.append(np.reshape(models[i].predict(fine_bez_input), (H,W,img_channels)))

mydir = "analysis/big_img_sep_models_analysis_"+str(m)+"_"+str(n)
if not os.path.exists(mydir):
    os.makedirs(mydir)


for i in range(n_img):
    bsplot(m, n, H, W, fine_preds[i], mydir+"/bsf_full_" + str(i) + "_" + str(m) + "_" + str(n) + ".png")
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(np.transpose(img_batch[i,:], [1,2,0]), cmap="gray")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(np.clip(preds[i],0,1), cmap="gray")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(np.abs(np.transpose(img_batch[i,:], [1,2,0])-np.clip(preds[i],0,1)), cmap="gray")
    plt.savefig(mydir+"/bsf_" + str(i) + "_" + str(m) + "_" + str(n) + ".png")
    plt.close()

