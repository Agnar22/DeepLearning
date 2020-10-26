import numpy as np
import matplotlib.pyplot as plt
import configparser
from tensorflow.keras.datasets import mnist
import time

from NeuralNetwork import Layers, Activations, Losses, Models, Regularizers


def cast_config_dict(config_dict):
    """

    :param config_dict:
    :return:
    """

    config_dict['layers'] = list(map(lambda x: int(x), config_dict['layers'].split(',')))
    config_dict['activations'] = config_dict['activations'].split(', ')
    config_dict['no_epochs'] = int(config_dict['no_epochs'])
    for key in ['learning_rate', 'l2_regularization']:
        config_dict[key] = float(config_dict[key])
    return config_dict


def read_config(path=None):
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


def load_data(file_path, to_one_hot, num_classes=None):
    """

    :param file_path:
    :param num_classes:
    :return:
    """

    data = np.genfromtxt(file_path, delimiter=',')
    x, y = data[:, :-1], data[:, -1]

    num_classes = int(num_classes if num_classes is not None else y.max() + 1)

    if to_one_hot:
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1

        return x, one_hot, num_classes
    else:
        return x, y.reshape(y.size, 1), num_classes


def create_model(units, activations, loss, lr, regularization):
    """

    :param units:
    :param activations:
    :param loss:
    :param lr:
    :param regularization:
    :return:
    """

    # Two different ways of creating the model
    model = Models.Sequential()
    x = Layers.Input(units.pop(0))
    # model.add(Layers.Input(units.pop(0)))
    for unit, activation in zip(units, activations):
        if activation == None:
            x = Layers.Dense(unit, activation=None, use_bias=True,
                             regularizer=Regularizers.L2(alpha=regularization))(x)
        # model.add(Layers.Dense(unit, activation=activation(), use_bias=True,
        #                        regularizer=Regularizers.L2(alpha=regularization)))
        else:
            x = Layers.Dense(unit, activation=activation(), use_bias=True,
                             regularizer=Regularizers.L2(alpha=regularization))(x)

    model.add(x)
    model.compile(loss=loss(), lr=lr)
    return model


def names_to_classes(classes, names):
    """

    :param classes:
    :param names:
    :return:
    """

    correct_classes = []
    for name in names:
        if name == 'NONE':
            correct_classes.append(Activations.Linear)
        for curr_class in classes:
            if curr_class().name == name:
                correct_classes.append(curr_class)
                break
    return correct_classes


def visualize(close, *args):
    """

    :param close:
    :param args:
    :return:
    """

    if close: plt.close('all')
    for func in args:
        plt.plot(func['x'], func['y'], label=func['name'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)

    config = read_config(path="config/config_ANY_STRUCTURE_CLASS.txt")

    x_train, y_train, num_classes = load_data(config['training'], config['loss_type'] == 'cross_entropy')
    x_val, y_val, _ = load_data(config['validation'], config['loss_type'] == 'cross_entropy', num_classes=num_classes)

    # # # # MNIST DATASET # # #
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28) / 255
    x_val = x_val.reshape(x_val.shape[0], 28 * 28) / 255
    y_train = np.array([[1 if x == y_train[n] else 0 for x in range(10)] for n in range(y_train.size)])
    y_val = np.array([[1 if x == y_val[n] else 0 for x in range(10)] for n in range(y_val.size)])

    activations = names_to_classes([Activations.ReLu, Activations.Linear, Activations.Tanh, Activations.Softmax],
                                   config['activations'])

    loss = names_to_classes([Losses.L2, Losses.Cross_Entropy], [config['loss_type']])[0]
    layers = [x_train.shape[-1]]
    layers.extend(config['layers'])
    if config['loss_type'] == 'cross_entropy':
        activations.append(Activations.Softmax)
        layers.append(num_classes)
    else:
        activations.append(Activations.Linear)
        layers.append(1)
    if layers[1] == 0:
        layers.pop(1)
        activations.pop(0)

    model = create_model(layers, activations, loss, config['learning_rate'], config['l2_regularization'])
    now = time.time()
    train_loss, val_loss = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config['no_epochs'],
                                     batch_size=64)
    visualize(True, {'x': list(range(len(train_loss))), 'y': train_loss, 'name': 'train_loss'},
              {'x': list(range(len(val_loss))), 'y': val_loss, 'name': 'val_loss'})
    z=model.predict(x_train)
    paths = model.save_model("Models", as_txt=True)
    # model.load_model(paths)
    #
    # # # # Visualize data # # #
    # import cv2
    #
    # for x in range(data.shape[0]):
    #     cv2.imshow(str(int(data[1970, -1])), val_data[1970, :784].reshape(28, 28))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
