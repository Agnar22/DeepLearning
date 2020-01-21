import Activations
import numpy as np


class Input:
    def __init__(self, size):
        self.units = size

    def forward(self, input):
        return input


class Dense:
    def __init__(self, units, activation=None, use_bias=True, kernel_regularizer=None):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.bias = np.zeros((units, 1))
        self.kernel_regularizer = kernel_regularizer
        self.input_size = None
        self.weighted_sum = None

    def __call__(self, inputs):
        self.prev_layer = inputs
        self.input_size = self.prev_layer.units
        # self.weights = np.random.uniform(low=-0.1, high=0.1, size=(self.input_size, self.units))
        self.weights = np.ones((self.input_size, self.units))
        return self

    def forward(self, input):
        # print("forward")
        inputs = self.prev_layer.forward(input)
        # print("inputs", inputs, inputs.shape)
        inputs = inputs.transpose()
        # print("weights", self.weights, self.weights.shape)
        self.weighted_sum = np.transpose(self.weights) @ inputs
        # print("weighted sum", self.weighted_sum, self.weighted_sum.shape)
        if self.use_bias:
            bias = np.repeat(self.bias, self.weighted_sum.shape[-1], axis=-1)
            # print("bias", bias, bias.shape, self.bias.shape)
            self.weighted_sum += bias
        if self.activation is None:
            return self.weighted_sum.transpose()
        return self.activation.forward(self.weighted_sum.transpose())

    def backward(self):
        pass


if __name__ == '__main__':
    # input = 7
    # layer = Dense(10)()
    # print(layer.next_layer)
    # x = np.array([[1, 2, 3]])
    # x = np.zeros((1, 10))
    # print(np.repeat(x, 6, axis=0))

    inp = Input(3)
    # a = Dense(1)(inp)
    # b = Dense(10)(inp)
    b = Dense(4)(inp)
    b = Dense(5)(b)
    b = Dense(6)(b)
    b = Dense(9)(b)
    outp = b.forward(np.array([[1, 10, 100], [-100, -100, -100]]))
    print(outp, outp.shape)
