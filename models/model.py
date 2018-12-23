from validation.validation import cross_entropy
import numpy as np


class Model:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        return layer

    def train(self, train_data, train_labels, epochs):
        for i in range(0, epochs):
            # forward pass
            x = train_data
            for layer in self.layers:
                x = layer.forward_pass(x)

            error = cross_entropy(train_labels, x)
            if i % 10 == 0:
                print("Loss: {}".format(error))

            da = -(np.divide(train_labels, x) - np.divide(1 - train_labels, 1 - x))
            for layer in reversed(self.layers):
                da = layer.backward_pass(da, self.learning_rate)

    def validate(self, test_data):
        x = test_data
        for layer in self.layers:
            x = layer.forward_pass(x)

        return x

    def save(self):
        pass

    def summary(self):
        for layer in self.layers:
            print(layer.summary())
