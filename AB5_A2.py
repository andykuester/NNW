# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

import tensorflow as tf

#%% TensorFlow command line parsing stuff
import argparse
FLAGS = None

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/data',
                       help='Directory for storing data')
FLAGS = parser.parse_args()

#%% Import data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#%% Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

#%% Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#%% Genauigkeit ausrechnen - nur für die Anzeige
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%% Der Zweck von sessions wird erst im Rahmen von Aufgabe 2 erklärt
sess = tf.InteractiveSession()

# Train
tf.initialize_all_variables().run()
for step in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if step % 20 == 0:
        print(step, sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

        #print (mnist.test.images.shape)

# Test trained model
print('result: ',sess.run(accuracy, feed_dict={x: mnist.test.images,
								  y_: mnist.test.labels}))


#%% Aufgabe 2a)
# Die Merkmale stehen in den Zeilen -> 784 Merkmale

#%% Aufgabe 2b)
# Die Batch Size ist der ausschlaggebende Wert für die Anzahl der Nachkommastellen
# Bei 100 Werten gibt es zum Beispiel 66 richtige Werte.
# --> 0.66 (Zwei Nachkommastellen)
# Bei 1000 Werten und 666 richtigen Werten
# --> 0.666 (Drei Nachkommastellen)

#%% Aufgabe 2c)
# Die letzte Ausgabe bezieht sich auf ein alle Test Daten
# Trainiert wird allerdings nicht mit allen Daten
# Aus diesem Grund ist die Genauigkeit aus den Trainingsdaten höher als 
# die von allen Daten
# 
