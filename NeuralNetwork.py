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
        total_bias_gradient = [np.zeros(b.shape) for b in self.biases]
        total_weight_gradient = [np.zeros(w.shape) for w in self.weights]
        for features, y in mini_batch:
            bias_gradient, weight_gradient = self.back_propagate(features, y)
            total_bias_gradient = add_lists(total_bias_gradient, bias_gradient)
            total_weight_gradient = add_lists(total_weight_gradient,
                                              weight_gradient)
        mini_batch_size = len(mini_batch)
        self.biases = gradient_descent_update(self.biases,
                                              total_bias_gradient,
                                              learning_rate, mini_batch_size)
        self.weights = gradient_descent_update(self.weights,
                                               total_weight_gradient,
                                               learning_rate, mini_batch_size)

    def back_propagate(self, features, y):
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

def add_lists(L1, L2):
    return [x1 + x2 for x1, x2 in zip(L1, L2)]

def gradient_descent_update(initial, gradient, learning_rate, num_examples):
    return [initial_x - (learning_rate / num_examples * derivative)
            for initial_x, derivative in zip(initial, gradient)]