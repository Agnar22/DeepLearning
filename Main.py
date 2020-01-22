import numpy as np
import matplotlib.pyplot as plt
import configparser
import enum
import sys

import NeuralNetwork as NN
from NeuralNetwork import Layers, Activations, Losses, Models


def cast_config_dict(config_dict):
    config_dict['layers'] = list(map(lambda x: int(x), config_dict['layers'].split(',')))
    config_dict['activations'] = config_dict['activations'].split(', ')
    for key in ['learning_rate', 'no_epochs', 'l2_regularization']:
        config_dict[key] = float(config_dict[key])
    return config_dict


def read_config(path="config.txt"):
    config = configparser.ConfigParser()
    config.read(path)
    config_dict = {}
    for key, value in config.items():
        for item in config[key]:
            print(item)
            config_dict[item] = config[key][item]
    return cast_config_dict(config_dict)


def create_model():
    # Special case if 0
    pass


def visualize():
    pass


def train_network(model, x_train, y_train, x_val, y_val):
    pass


if __name__ == '__main__':
    print(read_config())
    data = np.genfromtxt("Data/train_small.csv", delimiter=',')

    # import cv2
    #
    # for x in range(data.shape[0]):
    #     cv2.imshow(str(int(data[x, -1])), data[x, :784].reshape(28, 28))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    inp = Layers.Input(784)
    b = Layers.Dense(512, activation=Activations.ReLu(), use_bias=True)(inp)
    b = Layers.Dense(512, activation=Activations.ReLu(), use_bias=True)(b)
    b = Layers.Dense(512, activation=Activations.ReLu(), use_bias=True)(b)
    b = Layers.Dense(10, activation=Activations.Softmax(), use_bias=True)(b)
    model = NN.Models.Sequential()
    model.add(b)

    model.compile(loss=NN.Losses.L2(), lr=0.0000001)
    model.fit(data[:, :784],
              np.array([[1 if x == data[y, -1] else 0 for x in range(10)] for y in range(data.shape[0])]), epochs=200)
