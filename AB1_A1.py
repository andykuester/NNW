# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:48:33 2016

@author: a.kuester
"""


# Aufgabe 1

import numpy as np 

W=np.array([[1.,2,3],[3,4,5]])
W.shape
W.shape[0]
W.shape[1]

W.T
W.T.shape

type(W)
type(W.shape)

np.arange(10).shape

M=np.arange(12).reshape(3,4)
M
M.shape

M[2,0]
M[1,2]

M[1,:]
M[1]

M[:,1]
M[:,[1]]
M[:,[3,0,1,1]]
M[:,2:4]
M[-2,:]
M[-2:,:]
M[:,2]=2

M>4
M[M>4]
M[M>4]=-17
M

s=0
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        s+=M[i,j]
        print(s,',',sep='',end='')
s

def func(x):
    print(x*x)
    return x
    
func(3)

