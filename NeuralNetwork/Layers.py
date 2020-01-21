import Activations
import Losses
import numpy as np


# TODO: create kernel regularizer backward and forward

class Input:
    def __init__(self, size):
        self.units = size

    def forward(self, input):
        return input

    def backward(self, input):
        pass


class Dense:
    def __init__(self, units, activation=None, use_bias=True, kernel_regularizer=None):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.bias = np.zeros((units, 1))
        self.kernel_regularizer = kernel_regularizer
        self.input_size = None
        self.weighted_sum = None
        self.prev_layer_out = None

    def __call__(self, inputs):
        self.prev_layer = inputs
        self.input_size = self.prev_layer.units
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(self.input_size, self.units))
        # self.weights = np.ones((self.input_size, self.units))
        return self

    def forward(self, input):
        self.prev_layer_out = self.prev_layer.forward(input)
        self.weighted_sum = np.transpose(self.weights) @ self.prev_layer_out.transpose()
        if self.use_bias:
            bias = np.repeat(self.bias, self.weighted_sum.shape[-1], axis=-1)
            self.weighted_sum += bias
        if self.activation is None:
            return self.weighted_sum.transpose()
        return self.activation.forward(self.weighted_sum.transpose())

    def backward(self, temp_gradient):
        print("grad", temp_gradient.shape, self.weights.shape, self.prev_layer_out.shape)
        temp_gradient = self.activation.backward(temp_gradient) if self.activation is not None else temp_gradient

        delta_weights = self.prev_layer_out.transpose() @ temp_gradient
        delta_bias = temp_gradient
        # Update weights
        next_grad = np.transpose(self.weights @ np.transpose(temp_gradient))
        self.weights -= 0.01 * delta_weights
        self.prev_layer.backward(next_grad)


if __name__ == '__main__':
    inp = Input(1)
    # a = Dense(1)(inp)
    # b = Dense(10)(inp)
    b = Dense(40, activation=Activations.ReLu())(inp)
    # b = Dense(50, activation=Activations.ReLu())(b)
    # b = Dense(60, activation=Activations.ReLu())(b)
    # b = Dense(100, activation=Activations.ReLu())(inp)
    # b = Dense(10, activation=Activations.ReLu())(b)
    # b = Dense(100, activation=Activations.ReLu())(b)
    b = Dense(2, activation=Activations.Softmax())(b)
    for x in range(1000):
        outp = b.forward(np.array([[1], [-1]]))
        print(outp, outp.shape)
        # loss = Losses.L2()
        loss = Losses.Cross_Entropy()
        # print("loss", loss.forward(outp, np.array([[2], [10]])))
        # print("loss", loss.backward(outp, np.array([[0.1, 0.1, 0.1, 0.4], [0.4, 0.2, 1, 2]])))
        # b.backward(loss.backward(outp, np.array([[2], [10]])))
        # print("loss", loss.forward(outp, np.array([[2], [10]])))
        b.backward(loss.backward(outp, np.array([[0.0001, 0.9999], [0.9999, 0.0001]])))
        print("loss", loss.forward(outp, np.array([[0.0001, 0.9999], [0.9999, 0.0001]])))
