# -*- coding: utf-8 -*-

import tensorflow as tf

if __name__ == "__main__":
    tf.reset_default_graph()

    # 사칙연산
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

    # 행렬 연산
    with tf.device("/device:GPU:0"):
        x = tf.constant([[1.0, 2.0], [5.0, 6.0]])
        y = tf.eye(2)

        add = tf.add(x, y)  # x + y
        sub = tf.subtract(x, y)  # x - y
        mul = tf.multiply(x, y)  # x * y
        div = tf.div(x, y)  # x / y
        dot = tf.matmul(x, y)  # x . y

    with tf.Session() as sess:
        print(sess.run(add))  # [[2. 2.] [5. 7.]]
        print(sess.run(sub))  # [[0. 2.] [5. 5.]]
        print(sess.run(mul))  # [[1. 0.] [0. 6.]]
        print(sess.run(div))  # [[ 1. inf] [inf  6.]]
        print(sess.run(dot))  # [[1. 2.] [5. 6.]]
