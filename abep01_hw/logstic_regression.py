# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import datasets

if __name__ == "__main__":
    iris = datasets.load_iris()
    print(list(iris.keys()))
    X = iris["data"][:, 3:]  # 꽃잎의 너비
    Y = (iris["target"] == 2).astype(np.int)  # Iris-Virginica 면 1, 그렇지 않으면 0

    plt.plot(X[Y == 0], Y[Y == 0], 'ks')
    plt.plot(X[Y == 1], Y[Y == 1], 'rs')
    plt.show()
