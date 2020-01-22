import numpy as np


# TODO: Cross entropy backward correct?

class Loss:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def forward(self, prediction, target):
        return self.function(prediction, target).flatten().sum() / target.shape[0]

    def backward(self, prediction, target):
        return self.derivative(prediction, target)


class L2(Loss):
    def __init__(self):
        Loss.__init__(self, lambda x, y: (x - y) ** 2, lambda x, y: 2 * (x - y))


class Cross_Entropy(Loss):
    def __init__(self):
        Loss.__init__(self, lambda x, y: -y * np.log2(x), lambda x, y: 1 + (-y / (x)))  # * np.log(2)))


if __name__ == '__main__':
    pred = np.array([[0.15, 0.35, 0.25, 0.25], [0.05, 0.05, 0.9, 0.0000001]])
    target = np.array([[0, 0, 1, 0], [0, 0, 1, 0]])
    print(L2().forward(pred, target))
    print(L2().backward(pred, target))
    print(Cross_Entropy().forward(pred, target))
    print(Cross_Entropy().backward(pred, target))
