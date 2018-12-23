import numpy as np
from layers.layer import Dense, Input, Convolution, Flatten
from models.model import Model


def model(learning_rate, input_shape):
    mdl = Model(learning_rate)
    x = mdl.add(Dense(Input(input_shape), 4, 'relu', 'dense_1'))
    x = mdl.add(Dense(x, 2, 'relu', 'dense_2'))
    mdl.add(Dense(x, 1, 'sigmoid', 'out_1'))
    return mdl


def model(learning_rate, input_shape):
    mdl = Model(learning_rate)
    x = mdl.add(Convolution(Input(input_shape), (2, 10), 'relu', 'conv_1'))
    x = mdl.add(Convolution(x, (2, 10), 'relu', 'conv_2'))
    x = mdl.add(Flatten(x, name='flatten_1'))
    mdl.add(Dense(x, 1, 'sigmoid', 'out_1'))
    return mdl

# test_data = np.arange(1000) + 1000
# test_data = test_data.reshape((1000, 1))
#
# train_data = np.arange(1000)
# train_labels = (train_data % 2).reshape((1000, 1))
# train_data = train_data.reshape((1000, 1))
#
# mdl = model(0.05, (None, 1))
# mdl.summary()
# mdl.train(train_data, train_labels, 100000)
# print(mdl.validate(train_data))

x = np.arange(27).reshape((1, 3, 3, 3))
y = np.array([[1]])
mdl = model(0.05, (None, 3, 3, 3))
mdl.summary()
mdl.train(x, y, 10000)