# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:48:41 2016

@author: a.kuester
"""

#%% AB5
#%% Aufgabe 1
#%%

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
np.random.seed(42)
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3 + (np.random.randn(100).astype(np.float32)/40)

# Aufgabe 1g)
#y_data = np.cos(x_data) * 0.1 + 0.3 

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0, seed=42))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        
# Learns best fit is W: [0.1], b: [0.3]

#%% Aufgabe 1a)
Y=sess.run(y)
plt.scatter(x_data,y_data,marker='.')
plt.scatter(x_data,Y, marker='o',edgecolors='r',facecolors='none')
plt.show

#%% Aufgabe 1b)
# Es existiert ein Merkmal

#%% Aufgabe 1c)
# 0 [ 0.79622716] [-0.07830758]
# 20 [ 0.25390884] [ 0.22168453]
# 40 [ 0.12660754] [ 0.28609812]
# 60 [ 0.09728272] [ 0.30093628]
# 80 [ 0.09052755] [ 0.30435437]
# 100 [ 0.08897143] [ 0.30514175]
# ...
# 200 [ 0.08850598] [ 0.30537727]
#
# Es werden für jedes Eingabedatum zwei reelle Ausgabedaten erzeugt. 

#%% Aufgabe 1d)
# Regression

#%% Aufgabe 1e) 
# Summe der Fehlerquadrate

#%% Aufgabe 1f)
# 'x' is [[1., 1.]
#         [2., 2.]]
# tf.reduce_mean(x) ==> 1.5
# tf.reduce_mean(x, 0) ==> [1.5, 1.5]
# tf.reduce_mean(x, 1) ==> [1.,  2.]
# Es berechnet den Durchschnitt von verschiedenen Elementen 
