import numpy as np
import json
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import Model
import tensorflow as tf
from tensorflow.keras.optimizers import SGD


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
    max_target = max(targets)
    colours = [int(255 * x / max_target) for x in targets]
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
    shape = x.shape[1:]
    x = x.reshape((x.shape[0], -1))
    x = x / 255

    # Only using a subset of the data
    x, _, y, _ = train_test_split(x, y, test_size=1 - dss_size, random_state=42)

    # Dividing the data into subsets
    x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(x, y, test_size=unlabeled_size, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x_labeled, y_labeled, test_size=test_size, random_state=42)

    if one_hot:
        classes = max(max(y_train), max(y_test)) + 1
        y_train = np.array([to_one_hot(y_train[x], max=classes) for x in range(y_train.shape[0])])
        y_test = np.array([to_one_hot(y_test[x], max=classes) for x in range(y_test.shape[0])])
    return shape, (x_unlabeled, y_unlabeled), (x_labeled, y_labeled), (x_train, y_train), (x_test, y_test)


def setup_networks(params_autoencoder, params_classifier):
    autoencoder, encoder = Model.create_autoencoder(784, params_autoencoder)
    ssl = Model.create_classifier(params_classifier, encoder=encoder, freeze_encoder=False)
    classifier = Model.create_classifier(params_classifier, encoder=None, input_shape=784)
    encoder = tf.keras.Model(encoder[0], encoder[1])

    # TODO: use optimizer from parameters
    autoencoder.compile(optimizer=SGD(lr=params_autoencoder['lr'], momentum=0.9), loss=params_autoencoder['loss'])
    ssl.compile(optimizer=SGD(lr=params_classifier['lr'], momentum=0.9), loss=params_classifier['loss'],
                metrics=['accuracy'])
    classifier.compile(optimizer=SGD(lr=params_classifier['lr'], momentum=0.9), loss=params_classifier['loss'],
                       metrics=['accuracy'])

    return autoencoder, encoder, ssl, classifier


def train_network(network, x_train, y_train, val, batch_size=256, epochs=5, name="", metric='loss'):
    history = network.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=val).history

    train_loss = history[metric]
    val_loss = history['val_' + metric]
    train_graph = {'x': [x for x in range(len(train_loss))], 'y': train_loss, 'name': name + 'train'}
    val_graph = {'x': [x for x in range(len(val_loss))], 'y': val_loss, 'name': name + 'validation'}

    return train_graph, val_graph


if __name__ == '__main__':
    # TODO:
    # Network should scale number of I/O nodes on the fly
    # Use transposed convolutional layers
    # Freeze weights

    # INFO:
    # Dataset:
    #   -   D = entire dataset = DSS
    #   -   D1= unlabeled dataset
    #   -   D2= labeled dataset              D1+D2=D, D1>>D2
    #       - Split into train and validation
    #
    # # # # # Load parameters, set up data and create NNs # # # # #
    params = load_json("PivotalParameters.json")
    shape, (x_unlbl_train, y_unlbl_train), (x_unlbl_val, y_unlbl_val), (x_train, y_train), (x_val, y_val) = \
        load_data(params['dataset']['name'], params['dataset']['dssFraction'], params['dataset']['d1Fraction'],
                  params['dataset']['d2Training'], one_hot=True)
    autoencoder, encoder, ssl, classifier = setup_networks(params['autoencoder'], params['classifier'])

    # Training autoencoder on unlabeled data and plotting the loss
    autoencoder_train_graph, autoencoder_val_graph = train_network(autoencoder, x_unlbl_train, x_unlbl_train,
                                                                   (x_unlbl_val, x_unlbl_val), batch_size=8,
                                                                   epochs=params['autoencoder']['epochs'],
                                                                   name="autoencoder_")
    plot_graphs(False, autoencoder_train_graph, autoencoder_val_graph)

    # # # # # Display autoencoder reconstructions # # # # #
    display_images([x_unlbl_val[x:x + 1].reshape(shape) for x in range(params['display']['numReconstructions'])],
                   [autoencoder.predict(x_unlbl_val[x:x + 1]).reshape(shape) for x in
                    range(params['display']['numReconstructions'])],
                   label=y_unlbl_val)

    # # # # # Display TSNE of encoder output (latent vector) # # # # #
    if params['display']['show_tsne']:
        run_tsne(encoder.predict(x_unlbl_val[:1000]), y_unlbl_val[:1000].tolist())

    # # # # # Training the semi-supervised learned and the classifier, plotting the accuracy # # # # #
    ssl_train_graph, ssl_val_graph = train_network(ssl, x_train, y_train, (x_val, y_val), name="ssl_", metric='acc')
    classifier_train_graph, classifier_val_graph = train_network(classifier, x_train, y_train, (x_val, y_val),
                                                                 name="classifier_", metric='acc')
    plot_graphs(False, ssl_train_graph, ssl_val_graph, classifier_train_graph, classifier_val_graph)
