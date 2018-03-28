import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def bsplot(m, n, h, w, Z, fpath):
    fig=plt.figure(figsize=(20, 10))
    columns = Z.shape[2]
    rows = 3
    count = 1
    X = np.zeros((h,w)).astype(np.float32)
    Y = np.zeros((h,w)).astype(np.float32)
    for i in range(h):
        for j in range(w):
            X[i,j] = (0.5+i)/h
            Y[i,j] = (0.5+j)/w
    for k in range(columns):
        ax = fig.add_subplot(rows, columns, count)
        count += 1
        plt.imshow(Z[:,:,k], cmap="RdBu")
    for k in range(columns):
        ax = fig.add_subplot(rows, columns, count, projection='3d')
        count += 1
        surf = ax.plot_surface(X, Y, Z[:,:,k], rstride=1, cstride=1, 
                cmap="RdBu",linewidth=0, antialiased=False)
    fig.add_subplot(rows, columns, count)
    count += 1
    plt.imshow(np.clip(Z,0.,1.),cmap="gray")
    plt.savefig(fpath)
    plt.close()