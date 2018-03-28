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

def bez_mat(u, v, m, n):
    mCi = [0.]
    nCj = [0.]
    N = u.shape[0]
    bmat = np.zeros((N,1,m+1,n+1))
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
            bmat[:,0,i,j] = mCi[i] + nCj[j]
            bmat[:,0,i,j] += U[:,i,0] + U[:,m-i,1] + V[:,j,0] + V[:,n-j,1]
            bmat[:,0,i,j] = np.exp(bmat[:,0,i,j])
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

class BezierSurfaceFitter(Layer):
    def __init__(self, b, c, h, w, m, n, chunk_h=32, chunk_w=32, **kwargs):
        super(BezierSurfaceFitter, self).__init__(**kwargs)
        self.b = b
        self.c = c
        self.h = h
        self.w = w
        self.m = m
        self.n = n
        self.chunk_h = np.min([self.h,chunk_h])
        self.chunk_w = np.min([self.w,chunk_w])
    def build(self, input_shape):
        self.K_mat = []
        for i in range(self.b):
            K_mat_i = []
            for j in range(self.c):
                K_mat_i.append(self.add_weight(shape=(self.m+1, self.n+1),
                                    initializer="glorot_normal",
                                    name='kernel'))
            self.K_mat.append(K_mat_i)
        self.built = True
    def call(self, inputs):
        X_rec = []
        for i in range(self.b):
            X_rec.append([])
            for j in range(self.c):
                X_rec[i].append([])
        h_ = 0
        while h_ < self.h:
            c_h = np.min([self.h-h_, self.chunk_h])
            w_ = 0
            while w_ < self.w:
                c_w = np.min([self.w-w_, self.chunk_w])
                b_filter = bez_filter(self.h,self.w,self.m,self.n,h_,w_,c_h,c_w)
                for i in range(self.b):
                    for j in range(self.c):
                        my_output = K.reshape(K.sum(b_filter * self.K_mat[i][j], axis=(1,2,3)), (c_h,c_w))
                        if w_ == 0:
                            X_rec[i][j].append(my_output)
                        else:
                            X_rec[i][j][-1] = Concatenate(axis=1)([X_rec[i][j][-1],my_output])
                w_ += c_w
            h_ += c_h
        for i in range(self.b):
            for j in range(self.c):
                if len(X_rec[i][j]) > 1:
                    X_rec[i][j] = K.reshape(Concatenate(axis=0)(X_rec[i][j]), (1,1,1,self.h,self.w))
                else:
                    X_rec[i][j] = K.reshape(X_rec[i][j][0], (1,1,1,self.h,self.w))
            if self.c > 1:
                X_rec[i] = Concatenate(axis=2)(X_rec[i])
            else:
                X_rec[i] = X_rec[i][0]
        if self.b > 1:
            output = Concatenate(axis=1)(X_rec)
        else:
            output = X_rec[0]
        return output
    def compute_output_shape(self, input_shape):
        return (1, self.b, self.c, self.h, self.w)
    def get_config(self):
        base_config = super(BezierSurfaceFitter, self).get_config()
        return dict(list(base_config.items()))


def bs_build_model(b, c, h, w, m, n, lr=0.6):
    dummy_input = np.zeros((1, 1)).astype(np.float32)
    inputs = Input(shape=(1,))
    predictions = BezierSurfaceFitter(b,c,h,w,m,n)(inputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss="mean_squared_error",
                optimizer=optimizers.Adam(lr=lr))
    return model

def bs_fit(img_batch, m, n, lr=0.6, epochs=300, max_batch=1024):
    b,c,h,w = img_batch.shape
    if b > max_batch:
        print("Batch size bigger than max batch size:", b, "vs", max_batch)
        sys.exit(1)
    model = bs_build_model(b, c, h, w, m, n, lr)
    dummy_input = np.zeros((1, 1)).astype(np.float32)
    model.fit(dummy_input, np.reshape(img_batch,[1]+list(img_batch.shape)),
                batch_size=1,
                epochs=300,
                verbose=1)
    return model

def bs_get_weights(model):
    return model.layers[1].get_weights()

def bs_set_weights(model, K_mat):
    model.layers[1].set_weights(K_mat)