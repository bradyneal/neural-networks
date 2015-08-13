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

    def forward_propagate(self, input_layer, store_activations=False):
        a = column_vector(input_layer)
        activations = [a]
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w.dot(a) + b)
            if store_activations:
                activations.append(a)
        return activations if store_activations else a

    def mini_batch_gradient_descent(self, training_data, num_iterations,
                                    mini_batch_size, learning_rate,
                                    test_data=None):
        for iteration in range(num_iterations):
            mini_batches = randomly_partition(training_data, mini_batch_size)
            for mini_batch in mini_batches:
                self.mini_batch_update(mini_batch, learning_rate)
            if test_data is None:
                print("Iteration {0} of {1} complete...".format(
                    iteration, num_iterations))
            else:
                print("Iteration {0}: {1} / {2}".format(
                    iteration, self.evaluate(test_data), len(test_data)))

    def mini_batch_update(self, mini_batch, learning_rate):
        total_bias_gradient = [np.zeros(b.shape) for b in self.biases]
        total_weight_gradient = [np.zeros(w.shape) for w in self.weights]
        for features, y in mini_batch:
            activations = self.forward_propagate(features, True)
            bias_gradient, weight_gradient = self.back_propagate(activations, y)
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

    def back_propagate(self, activations, y):
        bias_gradient = [np.zeros(b.shape) for b in self.biases]
        weight_gradient = [np.zeros(w.shape) for w in self.weights]
        delta = activations[-1] - y
        bias_gradient[-1] = delta
        weight_gradient[-1] = delta.dot(activations[-2].T)
        # using negative indexing because len(activations) = len(weights) + 1
        for l in range(2, self.num_layers):
            sigmoid_prime = activations[-l] * (1 - activations[-l])
            delta = self.weights[-l + 1].T.dot(delta) * sigmoid_prime
            bias_gradient[-l] = delta
            weight_gradient[-l] = delta.dot(activations[-l - 1].T)
        return bias_gradient, weight_gradient

    def evaluate(self, test_data):
        return sum(int(self.predict(features) == y)
                   for features, y in test_data)

    def predict(self, input_layer):
        return np.argmax(self.forward_propagate(input_layer))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def column_vector(list0):
    arr = np.array(list0)
    return arr.reshape(arr.size, 1)

def randomly_partition(list0, partition_size):
    random.shuffle(list0)
    return (list0[i:i + partition_size] for
            i in range(0, len(list0), partition_size))

def add_lists(list1, list2):
    return [x1 + x2 for x1, x2 in zip(list1, list2)]

def gradient_descent_update(initial, gradient, learning_rate, num_examples):
    return [initial_x - (learning_rate / num_examples * derivative)
            for initial_x, derivative in zip(initial, gradient)]