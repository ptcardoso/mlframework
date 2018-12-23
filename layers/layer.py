import numpy as np
from numpy.fft import fft2, ifft2
from activations.activation import activation_factory
from models.model import Model


class Layer:
    def __init__(self, layer_input):
        self.layer_input = layer_input

    def forward_pass(self, x):
        pass

    def backward_pass(self, da, learning_rate):
        pass

    def summary(self):
        pass


class Input(Layer):
    def __init__(self, input_shape):
        super().__init__(None)
        self.shape = input_shape


class Dense(Layer):
    def __init__(self, layer_input, dimension, activation, name):
        """
        :param layer_input: object with shape property
        :param dimension: number of neurons in the fully connected layer
        :param activation: relu, sigmoid or tanh
        """
        super().__init__(layer_input)
        self.weights = np.random.rand(layer_input.shape[1], dimension) * 0.02
        self.activation = activation_factory(activation)
        self.bias = np.ones((1, dimension))
        self.name = name
        self.cache = None
        self.shape = (None, dimension)

    def forward_pass(self, x):
        z = np.dot(x, self.weights) + self.bias
        h = self.activation.apply(z)
        self.cache = (z, x, self.weights)
        return h

    def backward_pass(self, da, learning_rate):
        z, x, w = self.cache

        m = x.shape[0]
        dz = da * self.activation.derivative(z)
        dw = np.dot(x.T, dz) / m
        db = dz.sum(axis=1, keepdims=True) / m
        da_prev = np.dot(dz, w.T)

        self.weights = self.weights - learning_rate * dw
        self.bias = self.bias - learning_rate * db

        self.cache = None

        return da_prev

    def summary(self):
        return "Dense Layer %s: %s" % (self.name, self.shape)


class Convolution(Layer):
    def __init__(self, layer_input, kernel_dimensions, activation, name, stride=1):
        """
        :param layer_input: object with shape property
        :param kernel_dimensions: (a, b, c), a, b and c integers
        :param activation: 'relu', 'sigmoid'...
        :param name: layer name
        """
        super().__init__(layer_input)
        _, previous_h, previous_w, previous_c = layer_input.shape
        w, c = kernel_dimensions
        self.weights = np.random.rand(w, w, previous_c, c) * 0.02
        self.bias = np.ones((1, 1, 1, c))
        self.activation = activation_factory(activation)
        self.name = name
        self.stride = stride
        self.cache = None
        n_h = int((previous_h - w) / self.stride + 1)
        n_w = int((previous_w - w) / self.stride + 1)
        self.shape = (None, n_h, n_w, c)

    def conv_single_step(self, sample, weights, bias):
        z = np.multiply(sample, weights)
        z = np.sum(z)
        z += float(bias)
        return z

    def forward_pass(self, x):
        m, h_prev, w_prev, c_prev = x.shape
        f, f, c_prev, kernel_c = self.weights.shape  # assume kernel_h = kernel_w
        n_h = int((h_prev - f) / self.stride + 1)
        n_w = int((w_prev - f) / self.stride + 1)

        z = np.zeros((m, n_h, n_w, kernel_c))
        for i in range(0, m):
            sample = x[i]
            for h in range(0, n_h):
                for w in range(0, n_w):
                    for c in range(0, kernel_c):
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f

                        sample_slice = sample[vert_start:vert_end, horiz_start:horiz_end, :]
                        z[i, h, w, c] = self.conv_single_step(sample_slice, self.weights[:, :, :, c],
                                                              self.bias[:, :, :, c])

        activation = self.activation.apply(z)
        self.cache = (z, x, activation)
        return activation

    def backward_pass(self, da, learning_rate):
        z, x, activation = self.cache
        m, h_prev, w_prev, c_prev = x.shape
        f, f, c_prev, kernel_c = self.weights.shape  # assume kernel_h = kernel_w
        n_h = int((h_prev - f) / self.stride + 1)
        n_w = int((w_prev - f) / self.stride + 1)

        dz = da * self.activation.derivative(z)

        dw = np.zeros((f, f, c_prev, kernel_c))
        db = np.zeros((1, 1, 1, kernel_c))
        da_prev = np.zeros((m, h_prev, w_prev, c_prev))
        for i in range(0, m):
            da_prev_sample = da_prev[i]
            sample = x[i]
            for h in range(0, n_h):
                for w in range(0, n_w):
                    for c in range(0, kernel_c):
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f

                        da_prev_sample[vert_start:vert_end, horiz_start:horiz_end, :] += \
                            dz[i, h, w, c] * self.weights[:, :, :, c]
                        dw[:, :, :, c] += dz[i, h, w, c] * sample[vert_start:vert_end, horiz_start:horiz_end, :]
                        db[:, :, :, c] += dz[i, h, w, c]

        self.weights = self.weights - learning_rate * dw
        self.bias = self.bias - learning_rate * db

        return da_prev

    def summary(self):
        return "Convolutional Layer %s: %s" % (self.name, self.shape)


class Flatten(Layer):
    def __init__(self, layer_input, name):
        super().__init__(layer_input)
        self.name = name
        self.cache = None
        self.shape = self.get_flatten_dimensions(layer_input)

    def get_flatten_dimensions(self, x):
        cols = 1
        for idx in range(1, len(x.shape)):
            cols *= x.shape[idx]
        return x.shape[0], cols

    def forward_pass(self, x):
        self.cache = x
        flatten_dimensions = self.get_flatten_dimensions(x)
        return x.reshape(flatten_dimensions)

    def backward_pass(self, da, learning_rate):
        cache = self.cache
        self.cache = None
        return cache

    def summary(self):
        return "Flatten Layer %s: %s" % (self.name, self.shape)
