import numpy as np


class Activation:
    def apply(self, z):
        pass

    def derivative(self, z):
        pass


class Relu:
    def apply(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return (z > 0).astype(int)


class Sigmoid:
    def apply(self, z):
        return 1 / (1+np.exp(-z))

    def derivative(self, z):
        h = self.apply(z)
        return h*(1-h)


def activation_factory(activation_name):
    switcher = {
        'relu': Relu(),
        'sigmoid': Sigmoid()
    }
    return switcher.get(activation_name, 'Invalid activation')
