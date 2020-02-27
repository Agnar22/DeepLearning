import numpy as np
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2
import Model_tf


def load_json(filepath: str) -> json:
    with open(filepath) as f:
        return json.load(f)


def plot_graphs(close, *args):
    # TODO: what to visualize?
    # autoencoder training
    # semi-supervised learning and supervised training
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
    print(np.array(cases).shape)
    x_embedded = TSNE(n_components=2).fit_transform(np.array(cases))
    print(x_embedded.shape)
    print(type(x_embedded))
    print(targets)
    print(type(targets))
    colours = [100 * targets[x] for x in range(len(targets))]
    # plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=colours)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    for num, (color, group) in enumerate(zip(colours, targets)):
        x, y = x_embedded[num, :]
        print(num, x, y, group, color)
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()


def display_images(img1, img2):
    for num, (imgl, imgr) in enumerate(zip(img1, img2)):
        vis = np.concatenate((imgl, imgr), axis=1)
        cv2.imshow(str(num), vis)
    cv2.waitKey()


if __name__ == '__main__':
    # Notes:
    # - 4 Different datasets
    # - Only images
    # - Does not need to be optimal
    # TODO:
    # Network should scale number of I/O nodes on the fly
    # Use transposed convolutional layers
    # Modularity of autoencoder
    # Can have hidden layers after encoder
    print(load_json("config.json"))

    data = Model_tf.x_train
    # display_images([data[x].reshape(28, 28, 1) for x in range(10)], [data[y].reshape(28, 28, 1) for y in range(20, 30)])
    run_tsne(np.array([data[x].reshape(28, 28, 1) for x in range(1000)]).reshape(1000, 784),
             Model_tf.y_train[:1000].tolist())
