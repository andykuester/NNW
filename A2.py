# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:16:54 2016

@author: a.kuester
"""

import numpy as np 
import matplotlib.pyplot as plt

# Aufgabe 2a)
# 50 Beispiele je Iris Art

# Aufgabe 2b)

F=np.loadtxt('iris.csv','float','#',',')

X=F[:,0:4]
T=F[:,4]

Xt=X.T

plt.scatter(Xt[0],Xt[1])


plt.scatter(Xt[0],Xt[1],c=T,cmap=plt.cm.prism)