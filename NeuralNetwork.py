__author__ = 'brady'

import numpy as np
import random


class NeuralNetwork:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(size, previous_size) for
                        size, previous_size in zip(sizes[1:], sizes[:-1])]

    def forward_propagate(self, input_layer):
        a = column_vector(input_layer)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w.dot(a) + b)
        return a

    def mini_batch_gradient_descent(self, training_data, mini_batch_size,
                                    learning_rate, num_iterations):
        for iteration in range(num_iterations):
            mini_batches = randomly_partition(training_data, mini_batch_size)
            for mini_batch in mini_batches:
                self.mini_batch_update(mini_batch, learning_rate)

    def mini_batch_update(self, mini_batch, learning_rate):
        pass


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def column_vector(L):
    L = np.array(L)
    L.shape = (len(L), 1)
    return L

def randomly_partition(L, partition_size):
    random.shuffle(L)
    return (L[i:i + partition_size] for i in range(0, len(L), partition_size))