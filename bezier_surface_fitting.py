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
from keras.models import Model
from keras.engine import Layer
from keras import optimizers
from keras.layers import Concatenate, Input

# to prepare input to model
def bez_mat(u, v, m, n):
    mCi = [0.]
    nCj = [0.]
    N = u.shape[0]
    bmat = np.zeros((N,m+1,n+1))
    U = np.zeros((N,m+1,2))
    V = np.zeros((N,n+1,2))
    U[:,:,0] = np.dot(np.log(u).reshape((N,1)), np.arange(m+1).reshape((1,m+1)))
    U[:,:,1] = np.dot(np.log(1-u).reshape((N,1)), np.arange(m+1).reshape((1,m+1)))
    V[:,:,0] = np.dot(np.log(v).reshape((N,1)), np.arange(n+1).reshape((1,n+1)))
    V[:,:,1] = np.dot(np.log(1-v).reshape((N,1)), np.arange(n+1).reshape((1,n+1)))
    for i in range(1,m+1):
        if (m-i)<i:
            mCi.append(mCi[m-i])
        else:
            mCi.append(mCi[i-1] + np.log(m-i+1) - np.log(i))
    for j in range(1,n+1):
        if (n-j)<j:
            nCj.append(nCj[n-j])
        else:
            nCj.append(nCj[j-1] + np.log(n-j+1) - np.log(j))
    for i in range(m+1):
        for j in range(n+1):
            bmat[:,i,j] = mCi[i] + nCj[j]
            bmat[:,i,j] += U[:,i,0] + U[:,m-i,1] + V[:,j,0] + V[:,n-j,1]
            bmat[:,i,j] = np.exp(bmat[:,i,j])
    return bmat


def bez_filter(h,w,m,n,h_start=0,w_start=0,h_sz=None,w_sz=None):
    if h_sz is None:
        h_sz = h
    if w_sz is None:
        w_sz = w
    U = np.zeros(h_sz*w_sz)
    V = np.zeros(h_sz*w_sz)
    count = 0
    for i in range(h_start, h_start+h_sz):
        u = (i+0.5)/h
        for j in range(w_start, w_start+w_sz):
            v = (j+0.5)/w
            U[count] = u
            V[count] = v
            count += 1
    b_filter = bez_mat(U, V, m, n)
    return b_filter.astype(np.float32)

#####################

class BezierSurfaceFitter(Layer):
    def __init__(self, c, m, n, **kwargs):
        super(BezierSurfaceFitter, self).__init__(**kwargs)
        self.c = c
        self.m = m
        self.n = n
    def build(self, input_shape):
        self.K_mat = []
        for j in range(self.c):
            self.K_mat.append(self.add_weight(shape=(self.m+1, self.n+1),
                            initializer="glorot_normal",
                            name='kernel'))
        self.built = True
    def call(self, inputs):
        output = []
        for j in range(self.c):
            output.append(K.expand_dims(K.sum(inputs * self.K_mat[j], axis=(1,2))))
        if len(output) > 1:
            output = Concatenate(axis=1)(output)
        else:
            output = output[0]
        return output
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.c)
    def get_config(self):
        base_config = super(BezierSurfaceFitter, self).get_config()
        return dict(list(base_config.items()))


def bs_build_model(c, m, n, lr=0.6):
    inputs = Input(shape=(m+1,n+1))
    predictions = BezierSurfaceFitter(c,m,n)(inputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss="mean_squared_error",
                optimizer=optimizers.Adam(lr=lr))
    return model

def bs_fit(img, bez_input, m, n, lr=0.6, epochs=10, b_sz=1024):
    c,h,w = img.shape
    model = bs_build_model(c, m, n, lr)
    model.fit(bez_input, np.reshape(np.transpose(img, [1,2,0]), (h*w,c)),
                batch_size=b_sz,
                epochs=epochs,
                verbose=1)
    return model