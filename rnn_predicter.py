#coding:utf-8

import tensorflow as tf
import json
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
from get_trades import get_trades

date,price,amount =get_trades()

train_x=[]
train_y=[]
for i in range(0,20):
    #print(price)
    one_predictor=np.array(price[i:i+20],dtype=float)
    #print(one_predictor)
    train_x.append(one_predictor)
    if(int(price[i+20])>int(price[i+21])):
        train_y.append(np.array([1,0,0]))
    elif (int(price[i + 20]) == int(price[i + 21])):
        train_y.append(np.array([0,1,0]))
    elif(int(price[i+20])<int(price[i+21])):
        train_y.append(np.array([0,0,1]))

train_x=np.asarray(train_x,dtype=float)
train_x=np.resize(train_x,[20,20,1])
train_y=np.asarray(train_y)
print(train_y)

# Parameters
learning_rate = 0.001
training_iters = 1000000
batch_size = 20
display_step = 100

# Network Parameters
n_input = 1
n_steps = 20
n_hidden = 200
n_classes = 3

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = train_x,train_y
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        #correct_pred_value=sess.run(correct_pred, feed_dict={x: batch_x, y: batch_y})
        #print(correct_pred_value)
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
'''
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
'''

