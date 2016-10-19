# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:48:00 2016

@author: a.kuester
"""

import numpy as np

x=np.array([1.,2,3]) 
W=np.array([[1.,2,3],[4,3,2]])
W

x.shape
W.shape

W.dot(x)
W*W
#W.dot(W)

W.T
x.T
y=x[np.newaxis].T
y
y.shape

x2=np.array([[1.,2,3]])
x2.shape
y2=x2.T
y2
y2.shape

r1D=W.dot(x)
r1D.shape
r2D=W.dot(y)
r2D.shape

W.max()


class myclass:
    W=0
    _privateW=1
    def f(self):
        return self.W*2
    def __init__(self,W):
        self.W=W

m=myclass(np.array([-1,1]))
m.W
m.f()
