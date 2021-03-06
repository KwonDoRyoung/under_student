# -*- coding: utf-8 -*-

import numpy as np

if __name__ == "__main__":
    epochs = 2
    batch_size = 10
    inputs = np.arange(100)
    iter = len(inputs) // batch_size

    for epoch in range(epochs):
        print("{}th epoch".format(epoch))
        for n, step in enumerate(range(epoch * iter, (epoch + 1) * iter)):
            batch_x = inputs[n * batch_size:(n + 1) * batch_size]
            print("\t{}th: {}".format(step, batch_x))
