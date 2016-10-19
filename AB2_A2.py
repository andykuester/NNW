# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:08:23 2016

@author: a.kuester
"""

# AB2_A2

import numpy as np
import math as m
import nnwplot





class SLN:
    _threshold=2
    _dIn=0
    _cOut=0
    def __init__(self,dIn,cOut):
        self._b=np.zeros(cOut)
        self._b=self._b[np.newaxis].T
        
        np.random.seed(42)
        self._W=np.random.randn(dIn,cOut)/m.sqrt(dIn+1)
        
        
    def neuron(self,X):
        net=np.zeros(X.shape[1]) 
  
        net=self._W.dot(X) +self._b
             
        return net >= self._threshold
        
                
D=np.loadtxt('iris.csv','float','#',',')
DT=D.T
T=DT[4,:]
X=DT[0:4,:]
                   
        
nnw=SLN(1,X[0:2].shape[0]) # 1,2
result=nnw.neuron(X[0:2])

nnwplot.plotTwoFeatures(X[0:2],T,result)
        
    