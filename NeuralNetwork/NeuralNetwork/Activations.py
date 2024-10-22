import numpy as np


class Activation:
    def __init__(self, function, derivative):
        """

        :param function:
        :param derivative:
        """

        self.function = function
        self.derivative = derivative
        self.activations = None
        self.prev_layer = None

    def __call__(self, input: any) -> 'Activation':
        """

        :param input:
        :return:
        """

        self.prev_layer = input
        self.output_shape = self.prev_layer.output_shape
        return self

    def forward(self, input: np.ndarray) -> np.ndarray:
        """

        :param input:
        :return:
        """

        if self.prev_layer is not None:
            input = self.prev_layer.forward(input)
        self.activations = self.function(input)
        return self.activations

    def backward(self, temp_gradient: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient for this layer and sends it to the previous layer.
        Optimized by representing each jacobian as an array (since most activation
        functions have zeros on all non-diagonal entries) and thus takes the hadamard
        product instead of matrix multiplication.
        :param temp_gradient:
        :return:
        """

        gradient = self.derivative(self.activations) * temp_gradient
        if self.prev_layer is not None:
            return 0 + self.prev_layer.backward(gradient)
        else:
            return gradient


class ReLu(Activation):
    def __init__(self):
        Activation.__init__(self, lambda x: x * (x > 0), lambda x: x > 0)
        self.name = 'relu'


class Linear(Activation):
    def __init__(self):
        Activation.__init__(self, lambda x: x, lambda x: 1)
        self.name = 'linear'


class Tanh(Activation):
    def __init__(self):
        Activation.__init__(self, lambda x: np.tanh(x), lambda x: 1 - np.power(np.tanh(x), 2))
        self.name = 'tanh'


class Softmax(Activation):
    def __init__(self):
        """
        Using stable softmax
        :return:
        """

        Activation.__init__(self,
                            lambda x: np.exp(x - np.max(x, axis=-1, keepdims=True)) /
                                      np.exp(x - np.max(x, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True),
                            None)
        self.name = 'softmax'

    def backward(self, temp_gradient: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient for this layer and sends it to the previous layer.
        Cannot be optimized by representing each jacobian as an array, because
        there are non-zero values on non-diagonal entries
        :param temp_gradient:
        :return:
        """

        act_shape = self.activations.shape
        act = self.activations.reshape(act_shape[0], 1, act_shape[-1])

        jacobian = - (act.transpose((0, 2, 1)) @ act) * (1 - np.identity(self.activations.shape[-1]))
        jacobian += np.identity(act_shape[-1]) * (act * (1 - act)).transpose((0, 2, 1))

        gradient = (jacobian @ temp_gradient.reshape(act_shape[0], act_shape[-1], 1))
        gradient = gradient.reshape((act_shape[0], act_shape[-1]))

        if self.prev_layer is not None:
            self.prev_layer.backward(gradient)
        else:
            return gradient


if __name__ == '__main__':
    w_sum = np.array([[1, 3, -1], [1, -100, 3]])
    temp_grad = np.array([[0.3, -0.6, 1], [1, 1, 1]])
    r = Softmax()
    print(r.forward(w_sum))
    print(r.backward(temp_grad))
    print(w_sum.max(axis=-1))
