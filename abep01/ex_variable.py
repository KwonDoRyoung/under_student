# -*- coding: utf-8 -*-

import tensorflow as tf

# Monte Carlo Method
if __name__ == "__main__":
    tf.reset_default_graph()
    with tf.device("/device:GPU:0"):
        x = tf.random_uniform(shape=[], minval=0, maxval=1.0)
        y = tf.random_uniform(shape=[], minval=0, maxval=1.0)
        circle = tf.Variable(0, dtype=tf.float32, name="circle")

        up_circle = tf.assign(circle, circle + tf.constant(1.))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for count in range(10001):
            _x = sess.run(x)  # 0 to 1
            _y = sess.run(y)  # 0 to 1
            if _x * _x + _y * _y <= 1.:
                sess.run(up_circle)
            if count % 1000 == 0:
                circle_count = sess.run(circle)
                print("count: {}, pi: {:.4f}".format(count, (circle_count / count*1.0) * 4))
