# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# data 전처리
mnist = input_data.read_data_sets(".", one_hot=True)

# data 넣을 graph < 모델 (cnn)
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(dtype=tf.float32)
is_training = tf.placeholder(dtype=tf.bool)

# CNN 모델 part
c1 = tf.layers.conv2d(x, 32, (3, 3), padding="same", activation=None, use_bias=False)
bnc1 = tf.layers.batch_normalization(c1, training=is_training)
ac1 = tf.nn.relu(bnc1)
drop1 = tf.layers.dropout(ac1, keep_prob)

c2 = tf.layers.conv2d(drop1, 64, (3, 3), padding="same", activation=None, use_bias=False)
bnc2 = tf.layers.batch_normalization(c2, training=is_training)
ac2 = tf.nn.relu(bnc2)
drop2 = tf.layers.dropout(ac2, keep_prob)

p1 = tf.layers.max_pooling2d(drop2, (2, 2), strides=(2, 2))  # 14x14@64

c3 = tf.layers.conv2d(p1, 128, (3, 3), padding="same", activation=None, use_bias=False)
ac3 = tf.nn.relu(c3)
c4 = tf.layers.conv2d(ac3, 128, (3, 3), padding="same", activation=None, use_bias=False)
ac4 = tf.nn.relu(c4)

p2 = tf.layers.max_pooling2d(ac4, (2, 2), strides=(2, 2))  # 7x7@128

c5 = tf.layers.conv2d(p2, 256, (3, 3), padding="same", activation=None, use_bias=False)
ac5 = tf.nn.relu(c5)  # 7x7x256

flat = tf.layers.flatten(ac5)

fc1 = tf.layers.dense(flat, 256, use_bias=False)
afc1 = tf.nn.relu(fc1)
fc2 = tf.layers.dense(afc1, 10, use_bias=False)

# prediction part
predict = tf.nn.softmax(fc2)

# training part
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc2))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batch normalization을 위한 구문
with tf.control_dependencies(update_ops):
    opt = tf.train.AdamOptimizer(0.0001).minimize(cost)

correct = tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1)), tf.float32)
accuracy = tf.reduce_mean(correct)

# Session part
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        batch_x = np.reshape(batch_x, (-1, 28, 28, 1))
        _, loss = sess.run([opt, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8, is_training: True})

        if step % 100 == 0:
            test_x, test_y = mnist.test.next_batch(100)
            test_x = np.reshape(test_x, (-1, 28, 28, 1))
            acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1., is_training: False})
            print("{}th {:.4f}, {:.2f}".format(step, loss, acc))