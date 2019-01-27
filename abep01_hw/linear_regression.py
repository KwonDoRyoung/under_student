# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = 2 * np.random.rand(100, 1)
    Y = 8 - 2.5 * X + np.random.randn(100, 1)
    plt.plot(X, Y, 'ks')
    plt.title("y = 8 - 2.5*x")
    plt.show()
