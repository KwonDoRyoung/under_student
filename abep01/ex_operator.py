# -*- coding: utf-8 -*-

import tensorflow as tf

if __name__ == "__main__":
    tf.reset_default_graph()

    with tf.device("/device:GPU:0"):
        x = tf.constant(5.0)
        y = tf.constant(6.0)

        add = tf.add(x, y)
        sub = tf.subtract(x, y)
        mul = tf.multiply(x, y)
        div = tf.div(x, y)

    with tf.Session() as sess:
        print(sess.run(add))  # 11.0
        print(sess.run(sub))  # -1.0
        print(sess.run(mul))  # 30.0
        print(sess.run(div))  # 0.8333333
