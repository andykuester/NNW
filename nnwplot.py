# based on: https://github.com/eakbas/tf-svm/blob/master/plot_boundary_on_data.py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


def plotTwoFeatures(X,Y,pred_func):
    if X.ndim!=2:
        raise ValueError('X be a matrix (2 dimensional array).')
#    if X.shape[0]!=2: 
#        X=X.T
    if X.shape[0]!=2:
        raise ValueError('X must contain exactly 2 features.')
        
    # determine canvas borders
    mins = np.amin(X,1); 
    mins = mins - 0.1*np.abs(mins);
    maxs = np.amax(X,1); 
    maxs = maxs + 0.1*maxs;

    ## generate dense grid
    xs,ys = np.meshgrid(np.linspace(mins[0],maxs[0],300), 
            np.linspace(mins[1], maxs[1], 300));


    # evaluate model on the dense grid
    Z = pred_func(np.c_[xs.flatten(), ys.flatten()].T);
    Z = Z.reshape(xs.shape)

    # Plot the contour and training examples
    plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0,:], X[1,:], c=Y, s=50,
            cmap=colors.ListedColormap(['orange', 'blue', 'green']))
    plt.show()
