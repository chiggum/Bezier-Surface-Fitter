from bezier_surface_fitting import bez_mat, bez_filter

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import tensorflow as tf
sess = tf.Session()
from keras import backend as K

def z_func(bfltr, K_mat, m, n, h, w):
    output = K.reshape(K.conv2d(K.reshape(K_mat, (1,1,m+1,n+1)), bfltr), (h,w))
    output = sess.run(output)
    return output


def bsplot(m, n, h, w, bfltr, K_mat, fpath):
    fig=plt.figure(figsize=(20, 10))
    columns = len(K_mat)
    rows = 3
    count = 1
    X = np.zeros((h,w)).astype(np.float32)
    Y = np.zeros((h,w)).astype(np.float32)
    for i in range(h):
        for j in range(w):
            X[i,j] = (0.5+i)/h
            Y[i,j] = (0.5+j)/w
    Z = np.zeros((h,w,len(K_mat))).astype(np.float32)
    for k in range(len(K_mat)):
        Z[:,:,k] = z_func(bfltr, K_mat[k,:], m, n, h, w)
        ax = fig.add_subplot(rows, columns, count)
        count += 1
        plt.imshow(Z[:,:,k], cmap="RdBu")
    for k in range(len(K_mat)):
        ax = fig.add_subplot(rows, columns, count, projection='3d')
        count += 1
        surf = ax.plot_surface(X, Y, Z[:,:,k], rstride=1, cstride=1, 
                cmap="RdBu",linewidth=0, antialiased=False)
    fig.add_subplot(rows, columns, count)
    count += 1
    plt.imshow(np.clip(Z,0.,1.),cmap="gray")
    plt.savefig(fpath)
    plt.close()