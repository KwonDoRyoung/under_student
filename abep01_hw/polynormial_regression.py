# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = 8 * np.random.rand(100, 1) - 4
    Y = 2.7 - 1.75 * X + 0.42 * X * X + np.random.randn(100, 1)
    plt.plot(X, Y, 'ks')
    plt.title("y = 2.7 - 1.75*x + 0.42*x^2")
    plt.show()

    tf.reset_default_graph()

    with tf.device("/device:GPU:0"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="X")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Y")
        w2 = tf.Variable(initial_value=1., dtype=tf.float32, name="w2")
        w1 = tf.Variable(initial_value=-1., dtype=tf.float32, name="w1")
        b = tf.Variable(initial_value=0., dtype=tf.float32, name="b")
        pred = b + w1 * x + w2 * x * x
        losses = ((pred - y) ** 2) / 2
        loss = tf.reduce_mean(losses)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    n_epochs = 50
    batch_size = 10
    iter = len(X) // batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            print("{}th epoch".format(epoch))
            for i in range(iter):
                batch_x = X[i * batch_size:(i + 1) * batch_size]
                batch_y = Y[i * batch_size:(i + 1) * batch_size]
                l, _ = sess.run([loss, train_op], feed_dict={x: batch_x, y: batch_y})
                if i % 100 == 0:
                    print("loss: {:.4f}".format(l))
        weight2 = sess.run(w2)
        weight1 = sess.run(w1)
        bias = sess.run(b)

    plt.plot(X, Y, 'ks')
    X = np.arange(-4, 4)
    Y = bias + weight1 * X + weight2 * X * X
    plt.plot(X, Y, '-')
    plt.title("y = {:.4f} {:.4f}*x + {:.4f}*x^2".format(bias, weight1, weight2))
    plt.show()

