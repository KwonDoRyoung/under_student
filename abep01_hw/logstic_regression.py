# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import datasets

if __name__ == "__main__":
    iris = datasets.load_iris()
    print(list(iris.keys()))
    X = iris["data"][:, 3:]  # 꽃잎의 너비
    Y = (iris["target"] == 2).astype(np.int)  # Iris-Virginica 면 1, 그렇지 않으면 0

    plt.plot(X[Y == 0], Y[Y == 0], 'ks')
    plt.plot(X[Y == 1], Y[Y == 1], 'rs')
    plt.show()

    tf.reset_default_graph()

    Y = np.reshape(Y, (Y.shape[0], 1))
    with tf.device("/device:GPU:0"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="X")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Y")
        w = tf.Variable(initial_value=-1., dtype=tf.float32, name="w")
        b = tf.Variable(initial_value=0., dtype=tf.float32, name="b")
        pred = tf.sigmoid(b + w * x)
        losses = ((pred - y) ** 2) / 2
        loss = tf.reduce_mean(losses)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

    n_epochs = 100
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
        weight = sess.run(w)
        bias = sess.run(b)

    def sigmoid(x, derivative=False):
        return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

    plt.plot(X[Y == 0], Y[Y == 0], 'ks')
    plt.plot(X[Y == 1], Y[Y == 1], 'rs')
    X = np.arange(0, 2.7, 0.001)
    Y = sigmoid(bias + weight * X)
    plt.plot(X, Y, '-')
    plt.title("y = sigmoid({:.4f} + {:.4f}*x)".format(bias, weight))
    plt.show()


