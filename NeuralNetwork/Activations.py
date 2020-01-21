import numpy as np


class Activation:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
        self.activations = None

    def forward(self, input):
        self.activations = self.function(input)
        # print("act", self.activations.shape, input.shape)
        return self.activations

    def backward(self, temp_gradient):
        # print(self.derivative(self.activations).shape, temp_gradient.shape)
        return self.derivative(self.activations) * temp_gradient


class ReLu(Activation):
    def __init__(self):
        Activation.__init__(self, lambda x: x * (x > 0), lambda x: x > 0)


class Linear(Activation):
    def __init__(self):
        Activation.__init__(self, lambda x: x, lambda x: 1)


class Tanh(Activation):
    def __init__(self):
        Activation.__init__(self, lambda x: np.tanh(x), lambda x: 1 - np.power(np.tanh(x), 2))


class Softmax(Activation):
    def __init__(self):
        """
        Using stable softmax
        :return:
        """
        Activation.__init__(self,
                            lambda x: np.exp(x - np.max(x, axis=-1, keepdims=True)) /
                                      np.exp(x - np.max(x, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True),
                            lambda x: -np.power(np.e, x) * (1 - np.power(np.e, x))
                            )


if __name__ == '__main__':
    w_sum = np.array([[1, 3, -1], [1, -100, 3]])
    temp_grad = np.array([[0.3, -0.6, 1], [1, 1, 1]])
    r = Softmax()
    print(r.forward(w_sum))
    print(r.backward(temp_grad))
    print(w_sum.max(axis=-1))
