'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.5
training_epochs =5000
batch_size = 100
display_step = 50

# Network Parameters
n_hidden_1 = 500 # 1st layer number of features
#n_hidden_2 = 100 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    #'h1': tf.Variable(tf.zeros([n_input, n_hidden_1])), 
   'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1,seed=31)),
    #'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_classes],stddev=0.1,seed=31))
     'out': tf.Variable(tf.zeros([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):

        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x, batch_y = mnist.train.next_batch(100)
        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
        # Display logs per epoch step
        if epoch % display_step == 0:
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Epoch:", '%04d' % (epoch+1),accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
