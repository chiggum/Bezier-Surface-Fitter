from __future__ import print_function
import os
import sys
os.environ['KERAS_BACKEND'] = "tensorflow"
############################################
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
############################################
import keras
from keras.engine import Layer
from keras.layers import Concatenate

def bez_mat(u, v, m, n):
    mCi = [1.]
    nCj = [1.]
    bmat = np.zeros((m+1,n+1))
    U = np.ones((m+1,2))
    V = np.ones((n+1,2))
    for i in range(1,m+1):
        if (m-i)<i:
            mCi.append(mCi[m-i])
        else:
            mCi.append((mCi[i-1]*(m-i+1))/i)
        U[i,0] = u*U[i-1,0]
        U[i,1] = (1-u)*U[i-1,1]
    for j in range(1,n+1):
        if (n-j)<j:
            nCj.append(nCj[n-j])
        else:
            nCj.append((nCj[j-1]*(n-j+1))/j)
        V[j,0] = v*V[j-1,0]
        V[j,1] = (1-v)*V[j-1,1]
    for i in range(m+1):
        for j in range(n+1):
            bmat[i,j] = mCi[i]*nCj[j]*U[i,0]*U[m-i,1]*V[j,0]*V[n-j,1]
    return bmat


def bez_filter(c,h,w,m,n):
    b_filter = np.zeros((h*w,c,m+1,n+1))
    count = 0
    for i in range(h):
        u = (i+0.5)/h
        for j in range(w):
            v = (j+0.5)/w
            b_filter[count,:,:,:] = bez_mat(u,v,m,n)
            count += 1
    return b_filter.astype(np.float32)

class BezierSurfaceFitter(Layer):
    def __init__(self, b, c, h, w, m, n, **kwargs):
        super(BezierSurfaceFitter, self).__init__(**kwargs)
        self.b = b
        self.c = c
        self.h = h
        self.w = w
        self.m = m
        self.n = n
        self.b_filter = tf.transpose(bez_filter(1,h,w,m,n), [2,3,1,0])
    def build(self, input_shape):
        self.K_mat = [[]*self.c]*self.b
        for i in range(self.b):
            for j in range(self.c):
                self.K_mat[i][j] = self.add_weight(shape=(self.m+1, self.n+1),
                                    initializer="glorot_normal",
                                    name='kernel')
        self.built = True
    def call(self, inputs):
        X_rec = []
        for i in range(self.b):
            X_rec_i = []
            for j in range(self.c):
                X_rec_ = K.conv2d(K.reshape(self.K_mat[i][j], (1,1,self.m+1,self.n+1)),
                                self.b_filter)
                X_rec_i.append(K.reshape(X_rec_, (1,1,self.h,self.w)))
            X_rec.append(Concatenate(axis=1)(X_rec_i))
        return Concatenate(axis=0)(X_rec)
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        base_config = super(BezierSurfaceFitter, self).get_config()
        return dict(list(base_config.items()))


def bsfit(img, m, n, lr=0.6, epochs=300, max_batch=1024):
    b,c,h,w = img.shape
    if b > max_batch:
        print("Batch size bigger than max batch size:", b, "vs", max_batch)
        sys.exit(1)
    dummy_input = np.zeros((1, 1)).astype(np.float32)
    inputs = Input(shape=(1,))
    predictions.append(BezierSurfaceFitter(b,c,h,w,m,n)(inputs))
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss="mean_squared_error",
                optimizer=optimizers.Adam(lr=0.6))
    model.fit(dummy_input, img,
                batch_size=1,
                epochs=300,
                verbose=1)
    return model

img_rows = 32
img_cols = 32
img_channels = 3

(x_test, y_test), (x_train, y_train) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

img = x_train[0:1,:]
m = 15
n = 15
n_repeat = 1024
b_size = 1024

img_repeat = np.zeros((n_repeat, img_channels, img_rows, img_cols)).astype(img.dtype)
img_repeat[:,:,:,:] = img[0,:,:,:]

inputs = Input(shape=(img_channels,img_rows,img_cols))
predictions = BezierSurfaceFitter(img_channels, img_rows,img_cols,m,n)(inputs)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss="mean_squared_error",
              optimizer=optimizers.Adam(lr=0.6))
model.fit(img, img,
            batch_size=b_size,
            epochs=300,
            verbose=1)

pred = model.predict(img)
plt.imshow(np.transpose(img[0,:], [1,2,0]), cmap="gray")
plt.show()
plt.imshow(np.transpose(pred[0,:], [1,2,0]), cmap="gray")
plt.show()