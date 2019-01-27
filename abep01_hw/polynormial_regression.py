# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = 8 * np.random.rand(100, 1) - 4
    Y = 2.7 - 1.75 * X + 0.5 * X * X + np.random.randn(100, 1)
    plt.plot(X, Y, 'ks')
    plt.title("y = 2.7 - 1.75*x + 0.5*x^2")
    plt.show()
