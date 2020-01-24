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
            config_dict[item] = config[key][item]
    return cast_config_dict(config_dict)


def load_data(file_path, num_classes=None):
    """

    :param file_path:
    :param num_classes:
    :return:
    """

    data = np.genfromtxt(file_path, delimiter=',')
    x, y = data[:, :-1], data[:, -1]

    num_classes = num_classes if num_classes is not None else y.max() + 1
    one_hot = np.zeros((y.shape[0], num_classes.astype(int)))
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1

    return x, one_hot, num_classes


def create_model(units, activations, loss, lr, regularization):
    x = Layers.Input(units.pop(0))
    for unit, activation in zip(units, activations):
        x = Layers.Dense(unit, activation=activation, use_bias=True,
                         kernel_regularizer=Regularizers.L2(alpha=regularization))(x)

    model = Models.Sequential()
    model.add(x)
    model.compile(loss=loss, lr=lr)
    return model


def visualize(close, *args):
    if close: plt.close('all')
    for func in args:
        plt.plot(func['x'], func['y'], label=func['name'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)

    x_train, y_train, num_classes = load_data("Data/train_small.csv")
    x_val, y_val, _ = load_data("Data/validate_small.csv")

    # import cv2
    #
    # for x in range(data.shape[0]):
    #     cv2.imshow(str(int(data[1970, -1])), val_data[1970, :784].reshape(28, 28))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    inp = Layers.Input(784)
    b = Layers.Dense(512, activation=Activations.ReLu(), use_bias=True, kernel_regularizer=Regularizers.L1())(inp)
    b = Layers.Dense(512, activation=Activations.ReLu(), use_bias=True, kernel_regularizer=Regularizers.L2())(b)
    b = Layers.Dense(512, activation=Activations.ReLu(), use_bias=True, kernel_regularizer=Regularizers.L2())(b)
    b = Layers.Dense(10, activation=Activations.Softmax(), use_bias=True, kernel_regularizer=Regularizers.L2())(b)
    # b = Activations.Softmax()(b)
    model = NN.Models.Sequential()
    model.add(b)

    model.compile(loss=NN.Losses.Cross_Entropy(), lr=0.01)
    # print(model.save_model("Models"))
    # (x_train, y_train), (x_val, y_val) = mnist.load_data()
    # x_train = x_train.reshape(x_train.shape[0], 28 * 28) / 255
    # x_val = x_val.reshape(x_val.shape[0], 28 * 28) / 255
    #
    # model.fit(x_train,
    #           np.array([[1 if x == y_train[y] else 0 for x in range(10)] for y in range(y_train.shape[0])]),
    #           validation_data=(x_val, np.array(
    #               [[1 if x == y_val[y] else 0 for x in range(10)] for y in range(y_val.shape[0])])), epochs=200)
    #
    train_loss, val_loss = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)
    visualize(True, {'x': list(range(len(train_loss))), 'y': train_loss, 'name': 'train_loss'},
              {'x': list(range(len(val_loss))), 'y': val_loss, 'name': 'val_loss'})
    # print(model.predict(x_val, y_val))
    # print(model.predict(x_val, y_val))
    # paths = model.save_model("Models", as_txt=True)
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1)
    # model.load_model(paths)
    # print(model.predict(x_val, y_val))
    # print(model.predict(x_val, y_val))
    # print(model.predict(x_val, y_val))
