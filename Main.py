import numpy as np
import matplotlib.pyplot as plt
import configparser
import enum
import sys

import NeuralNetwork as NN


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