import numpy as np
import matplotlib.pyplot as plt
import configparser
import enum
import sys
from tensorflow.keras.datasets import mnist

import NeuralNetwork as NN
from NeuralNetwork import Layers, Activations, Losses, Models, Regularizers


def cast_config_dict(config_dict):
    """

    :param config_dict:
    :return:
    """

    config_dict['layers'] = list(map(lambda x: int(x), config_dict['layers'].split(',')))
    config_dict['activations'] = config_dict['activations'].split(', ')
    for key in ['learning_rate', 'no_epochs', 'l2_regularization']:
        config_dict[key] = float(config_dict[key])
    return config_dict


def read_config(path="config.txt"):
    """

    :param path:
    :return:
    """

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
    np.random.seed(42)

    data = np.genfromtxt("Data/train_small.csv", delimiter=',')
    val_data = np.genfromtxt("Data/validate_small.csv", delimiter=',')

    # import cv2
    #
    # for x in range(data.shape[0]):
    #     cv2.imshow(str(int(data[1970, -1])), val_data[1970, :784].reshape(28, 28))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    inp = Layers.Input(784)
    b = Layers.Dense(512, activation=Activations.ReLu(), use_bias=True, kernel_regularizer=Regularizers.L2())(inp)
    b = Layers.Dense(256, activation=Activations.ReLu(), use_bias=True, kernel_regularizer=Regularizers.L2())(b)
    b = Layers.Dense(128, activation=Activations.ReLu(), use_bias=True, kernel_regularizer=Regularizers.L2())(b)
    b = Layers.Dense(10, activation=Activations.Softmax(), use_bias=True, kernel_regularizer=Regularizers.L2())(b)
    # b = Activations.Softmax()(b)
    model = NN.Models.Sequential()
    model.add(b)

    model.compile(loss=NN.Losses.Cross_Entropy(), lr=0.0000001)
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28) / 255
    x_val = x_val.reshape(x_val.shape[0], 28 * 28) / 255

    model.fit(x_train,
              np.array([[1 if x == y_train[y] else 0 for x in range(10)] for y in range(y_train.shape[0])]),
              validation_data=(x_val, np.array(
                  [[1 if x == y_val[y] else 0 for x in range(10)] for y in range(y_val.shape[0])])), epochs=20)
    model.fit(data[:, :784],
              np.array([[1 if x == data[y, -1] else 0 for x in range(10)] for y in range(data.shape[0])]),
              validation_data=(val_data[:, :784], np.array(
                  [[1 if x == val_data[y, -1] else 0 for x in range(10)] for y in range(data.shape[0])])), epochs=200)
