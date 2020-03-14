import numpy as np
import json
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import Model
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam


def load_json(filepath: str) -> json:
    with open(filepath) as f:
        return json.load(f)


def plot_graphs(close, *args):
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


def run_tsne(cases, targets):
    x_embedded = TSNE(n_components=2, perplexity=50).fit_transform(np.array(cases))
    max_target = max(targets)[0]
    colours = [int(255 * x[0] / max_target) for x in targets]
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=colours)
    plt.show()


def display_images(img1, img2, label=None):
    for num, (imgl, imgr) in enumerate(zip(img1, img2)):
        vis = np.concatenate((imgl, imgr), axis=1)
        if label is None:
            cv2.imshow(str(num), vis)
        else:
            cv2.imshow(str(label[num]), vis)
    cv2.waitKey()


def to_one_hot(num, max=10):
    return [1 if x == num else 0 for x in range(max)]


def get_data(name):
    if name == 'mnist':
        return tf.keras.datasets.mnist.load_data()
    elif name == 'fashion_mnist':
        return tf.keras.datasets.fashion_mnist.load_data()
    elif name == 'cifar10':
        return tf.keras.datasets.cifar10.load_data()
    elif name == 'cifar100':
        return tf.keras.datasets.cifar100.load_data()


def load_data(name, dss_size, unlabeled_size, test_size, one_hot=False):
    (x_train, y_train), (x_test, y_test) = get_data(name)
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    if len(x.shape) == 3:
        x = x.reshape((*x.shape, 1))
    input_shape = x.shape[1:]
    output_shape = y.shape[1:]
    x = x / 255

    # Only using a subset of the data
    x, _, y, _ = train_test_split(x, y, test_size=1 - dss_size, random_state=42)

    # Dividing the data into subsets
    x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(x, y, test_size=unlabeled_size, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x_labeled, y_labeled, test_size=test_size, random_state=42)

    if one_hot:
        classes = int(max(max(y_train), max(y_test)) + 1)
        y_train = np.array([to_one_hot(int(y_train[x]), max=classes) for x in range(y_train.shape[0])])
        y_test = np.array([to_one_hot(int(y_test[x]), max=classes) for x in range(y_test.shape[0])])
        output_shape = y_train.shape[1:]
    return input_shape, output_shape, (x_unlabeled, y_unlabeled), (x_labeled, y_labeled), (x_train, y_train), \
           (x_test, y_test)


def get_optimizer(optim_name, lr):
    if optim_name == "sgd":
        return SGD(learning_rate=lr, momentum=0.9)
    elif optim_name == "rmsprop":
        return RMSprop(learning_rate=lr)
    elif optim_name == "adagrad":
        return Adagrad(learning_rate=lr)
    return Adam(learning_rate=lr)


def setup_networks(params_autoencoder, params_classifier, input_shape, output_shape):
    autoencoder, encoder, decoder = Model.create_autoencoder(input_shape, params_autoencoder)
    ssl = Model.create_classifier(latent_size=params_autoencoder['latentSize'], encoder=encoder,
                                  input_shape=input_shape, output_shape=output_shape)
    classifier = Model.create_classifier(latent_size=params_autoencoder['latentSize'], input_shape=input_shape,
                                         output_shape=output_shape)

    autoencoder.compile(
        optimizer=get_optimizer(params_autoencoder['optimizer'], params_autoencoder['lr']),
        loss=params_autoencoder['loss'])
    classifier.compile(
        optimizer=get_optimizer(params_classifier['optimizer'], params_classifier['lr']),
        loss=params_classifier['loss'], metrics=['accuracy'])

    return autoencoder, encoder, decoder, ssl, classifier


def train_network(network, x_train, y_train, val, batch_size=64, epochs=5, name="", metric='loss'):
    history = network.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=val).history

    train_loss = history[metric]
    val_loss = history['val_' + metric]
    train_graph = {'x': [x for x in range(len(train_loss))], 'y': train_loss, 'name': name + 'train'}
    val_graph = {'x': [x for x in range(len(val_loss))], 'y': val_loss, 'name': name + 'validation'}

    return train_graph, val_graph


def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False


if __name__ == '__main__':
    # Dataset:
    #   -   D = entire dataset
    #   -   DSS = fraction of D
    #   -   D1= unlabeled dataset
    #   -   D2= labeled dataset              D1+D2=DSS, D1>>D2

    # # # # # Load parameters, set up data and create NNs # # # # #
    params = load_json("PivotalParameters.json")
    input_shape, output_shape, (x_unlbl_train, y_unlbl_train), (x_unlbl_val, y_unlbl_val), \
    (x_train, y_train), (x_val, y_val) = load_data(params['dataset']['name'], params['dataset']['dssFraction'],
                                                   params['dataset']['d1Fraction'], 1-params['dataset']['d2Training'],
                                                   one_hot=True)
    autoencoder, encoder, _, ssl, classifier = setup_networks(params['autoencoder'], params['classifier'],
                                                              input_shape, output_shape)

    # # # # # Display TSNE of encoder output (latent vector) before unsupervised training # # # # #
    if params['display']['show_tsne']:
        run_tsne(encoder.predict(x_unlbl_val[:1000]), y_unlbl_val[:1000].reshape(-1, 1))

    # Training autoencoder on unlabeled data and plotting the loss
    autoencoder_train_graph, autoencoder_val_graph = train_network(autoencoder, x_unlbl_train, x_unlbl_train,
                                                                   (x_unlbl_val, x_unlbl_val), batch_size=8,
                                                                   epochs=params['autoencoder']['epochs'],
                                                                   name="autoencoder_")
    plot_graphs(False, autoencoder_train_graph, autoencoder_val_graph)

    # # # # # Freezing encoder and compiles semi-supervised learner
    if params['autoencoder']['freezeEncoder']:
        freeze_model(encoder)

    ssl.compile(optimizer=get_optimizer(params['classifier']['optimizer'], params['classifier']['lr']),
                loss=params['classifier']['loss'], metrics=['accuracy'])

    # # # # # Display autoencoder reconstructions # # # # #
    display_images([x_unlbl_val[x:x + 1].reshape(input_shape) for x in range(params['display']['numReconstructions'])],
                   [autoencoder.predict(x_unlbl_val[x:x + 1]).reshape(input_shape) for x in
                    range(params['display']['numReconstructions'])],
                   label=y_unlbl_val)

    # # # # # Display TSNE of encoder output (latent vector) after unsupervised training # # # # #
    if params['display']['show_tsne']:
        run_tsne(encoder.predict(x_unlbl_val[:1000]), y_unlbl_val[:1000].reshape(-1, 1))

    # # # # # Training the semi-supervised learned and the classifier, plotting the accuracy # # # # #
    ssl_train_graph, ssl_val_graph = train_network(ssl, x_train, y_train, (x_val, y_val),
                                                   epochs=params['classifier']['epochs'], name="ssl_", metric='acc')
    classifier_train_graph, classifier_val_graph = train_network(classifier, x_train, y_train, (x_val, y_val),
                                                                 epochs=params['classifier']['epochs'],
                                                                 name="classifier_", metric='acc')
    plot_graphs(False, ssl_train_graph, ssl_val_graph, classifier_train_graph, classifier_val_graph)

    # # # # # Display TSNE of encoder output (latent vector) after supervised training # # # # #
    if params['display']['show_tsne']:
        run_tsne(encoder.predict(x_unlbl_val[:1000]), y_unlbl_val[:1000].reshape(-1, 1))

