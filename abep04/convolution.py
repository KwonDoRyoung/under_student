# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# hyper parameter
batch_size = 100

# data call
datasets = input_data.read_data_sets(".", one_hot=True)

# CNN 모델 구조 - 입력
inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="inputs")
labels = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")

# CNN 모델
c1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], strides=[1, 1],
                      padding="same", activation=tf.nn.relu, name="conv1")  # batch x 28 x 28 @ 32
p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2], strides=[2, 2], padding="valid",
                             name="pool1")  # batch x 14 x 14 @ 32
c2 = tf.layers.conv2d(inputs=p1, filters=64, kernel_size=[3, 3], strides=[1, 1],
                      padding="same", activation=tf.nn.relu, name="conv2")  # batch x 14 x 14 @ 64
p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2], strides=[2, 2], padding="valid",
                             name="pool2")  # batch x 7 x 7 @ 64

# CNN Feature 로부터 Classification MLP
flat = tf.layers.flatten(inputs=p2)
fc1 = tf.layers.dense(inputs=flat, units=1000, activation=tf.nn.relu, name="fc1")
output = tf.layers.dense(inputs=flat, units=10, name="output")

predict = tf.nn.softmax(output)
correct = tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(predict, 1)), tf.float32)
accuracy = tf.reduce_mean(correct)

with tf.name_scope("cost"):
    costs = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=output)
    cost = tf.reduce_mean(costs)

opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train
    for iter in range(10000):
        batch_x, batch_y = datasets.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
        c, _ = sess.run([cost, opt], feed_dict={inputs: batch_x, labels: batch_y})

        if iter % 100 == 0:
            valid_x, valid_y = datasets.validation.next_batch(batch_size)
            valid_x = np.reshape(valid_x, [-1, 28, 28, 1])
            acc = sess.run(accuracy, feed_dict={inputs: valid_x, labels: valid_y})
            print("{}th loss: {:.4f}, valid accuracy: {:.2f}".format(iter, c, acc))

    # test
    test_x, test_y = datasets.test.next_batch(batch_size)
    test_x = np.reshape(test_x, [-1, 28, 28, 1])
    acc = sess.run(accuracy, feed_dict={inputs: test_x, labels: test_y})
    print("test accuracy: {:.2f}".format(acc))
