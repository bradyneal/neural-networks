__author__ = 'brady'

import numpy as np


class NeuralNetwork:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(size, previous_size) for
                        size, previous_size in zip(sizes[1:], sizes[:-1])]

    def forward_propogate(self, input):
        a = to_column_vector(input)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w.dot(a) + b)
        return a


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def to_column_vector(l):
    l = np.array(l)
    l.shape = (len(l), 1)
    return l