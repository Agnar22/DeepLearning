import numpy as np
import json
from sklearn.manifold import TSNE


def load_json(filepath: str) -> json:
    with open(filepath) as f:
        return json.load(f)


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
