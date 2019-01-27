# -*- coding: utf-8 -*-

import tensorflow as tf

if __name__ == "__main__":
    tf.reset_default_graph()
    with tf.device("/device:GPU:0"):
        pi = tf.constant(value=3.141592, dtype=tf.float32, shape=[], name="PI")
        r = tf.placeholder(dtype=tf.float32, shape=[2, 1], name="radius")

        r_2 = r * r  # r^2
        area = tf.multiply(pi, r_2)  # pi * r^2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # circle_area = sess.run(area)  # code error: feed_dict 필수
        circle_area = sess.run(area, feed_dict={r: [[1.5], [3]]})
        print(circle_area)  # [[ 7.068582] [28.274328]]
