# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:08:23 2016

@author: a.kuester
"""

# AB2_A2

import numpy as np
import math as m
import nnwplot

def ErrorRate(self,Y,T):
    if  Y.ndim==1 or Y.shape[0]==1:
        errors=Y!=T
        return errors.sum()/Y.size
    else: #für mehrere Ausgaben in one-hot Kodierung:
        errors=Y.argmax(0)!=T.argmax(0)
        return errors.sum()/Y.shape[1]



class SLN:
    _threshold=0
    _dIn=0
    _cOut=0
    _W=0
    def __init__(self,dIn,cOut):
        self._b=np.zeros(cOut)[np.newaxis].T
        np.random.seed(42)
        self._W=np.random.randn(cOut,dIn)/np.sqrt(dIn)
    def neuron(self,X):
        net=self._W.dot(X) +self._b        
        return net >= self._threshold  
    
        
    def DeltaTrain(self,X,T,eta,maxIter,maxErrorRate):

        i = 0
        err=1

        N=X.shape[1]    # Anzahl Trainingsdaten    
        x0 = np.ones(N)[np.newaxis]
        
        while (i < maxIter and (err > maxErrorRate)):
            
            Y = self.neuron(X)
            
            #err = self.ErrorRate(y,T)
            err = np.linalg.norm( T - y)
            
            self._W = self._W +( (eta*((T - y).dot(X.T))) / float(T.shape[0]))
            self._b = self._b + ((T - y) / float(T.shape[0]))
            
            self._W+=eta*(T-Y).dot(X.T)/N
            self._b+=eta*(T-Y).dot(x0.T)/N            
            
        
       
 
        
iris = np.loadtxt("iris.csv",delimiter=',')
X=iris[:,0:4].T
T=iris[:,4]   
                         
        
nnw=SLN(2,1) # 2,1
nnw._W=np.array([-1,1])
nnw._b=np.array([3])
#result=nnw.neuron(X)
eta=0.1
maxIter=10
maxErrorRate=0.01
result=nnw.DeltaTrain(X,T,eta,maxIter,maxErrorRate)

    