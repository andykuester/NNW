# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:41:32 2016

@author: a.kuester
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors

# Vorgabe

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


# Aus Aufgabe 2
F=np.loadtxt('iris.csv','float','#',',')
X=F[:,0:4]
T=F[:,4]
Xt=X.T



# Aufgabe 3
threshold=2

def neuron(X):
    net=np.zeros(X.shape[:1])    
    W=[-0.3,1]
    #W=[-0.2,1]
    #W=[-0.1,1]
    #W=[0,1]    
    M1=X[:,0:1]
    M2=X[:,1:2]
    for i in range(net.shape[0]):
        net[i]=(M1[i]*W[0])+(M2[i]*W[1])
    return net  
    
result=neuron(X)    

final=result>=threshold
        
     
     
     
     
     
     
     
     
     
     
     
     
     