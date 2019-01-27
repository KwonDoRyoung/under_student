# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_california_housing

if __name__ == "__main__":
    tf.reset_default_graph()

    with tf.device("/device:GPU:0"):
        scalar = tf.Variable(initial_value=3, dtype=tf.float32, name="scalar")
        vector = tf.Variable(initial_value=[3], dtype=tf.float32, name="vector")
        matrix_2d = tf.Variable(initial_value=[[3], [2]], dtype=tf.float32, name="matrix_2d")
        matrix_3d = tf.Variable(initial_value=[[[3], [2]], [[10], [11]]], dtype=tf.float32, name="matrix_3d")

        scalar_rank, scalar_shape = tf.rank(scalar), tf.shape(scalar)
        vector_rank, vector_shape = tf.rank(vector), tf.shape(vector)
        matrix_2d_rank, matrix_2d_shape = tf.rank(matrix_2d), tf.shape(matrix_2d)
        matrix_3d_rank, matrix_3d_shape = tf.rank(matrix_3d), tf.shape(matrix_3d)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run([scalar_rank, scalar_shape]))  # [0, array([], dtype=int32)]
        print(sess.run([vector_rank, vector_shape]))  # [1, array([1], dtype=int32)]
        print(sess.run([matrix_2d_rank, matrix_2d_shape]))  # [2, array([2, 1], dtype=int32)]
        print(sess.run([matrix_3d_rank, matrix_3d_shape]))  # [3, array([2, 2, 1], dtype=int32)]
