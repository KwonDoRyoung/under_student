# -*- coding: utf-8 -*-

import tensorflow as tf

if __name__ == "__main__":
    tf.reset_default_graph()
    with tf.device("/device:GPU:0"):
        b = tf.placeholder(dtype=tf.bool, shape=[2], name="Bool")
        i = tf.placeholder(dtype=tf.int32, shape=[2], name="Int")
        f = tf.placeholder(dtype=tf.float32, shape=[1, 2], name="Float")
        s = tf.placeholder(dtype=tf.string, shape=[], name="String")

    with tf.Session() as sess:
        # p_b = sess.run(b)  # error code
        sess.run(tf.global_variables_initializer())
        p_b = sess.run(b, feed_dict={b: [False, True]})
        p_i = sess.run(i, feed_dict={i: [5, 6]})
        p_f = sess.run(f, feed_dict={f: [[0.444, 9.273]]})
        p_s = sess.run(s, feed_dict={s: "Welcome, Tensorflow World!"})
        print(p_b)  # [False  True]
        print(p_i)  # [5 6]
        print(p_f)  # [[0.444 9.273]]
        print(p_s)  # Welcome, Tensorflow World!
