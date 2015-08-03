__author__ = 'brady'

import numpy as np


class NeuralNetwork:

    def __init__(self, sizes, input=None):
        assert input is None and sizes[0] == len(input), \
            "Input size not compatible with sizes list"
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.input = input
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(size, previous_size) for
                        size, previous_size in zip(size[1:], size[:-1])]

    def forward_propogate(self, input=None):
        
