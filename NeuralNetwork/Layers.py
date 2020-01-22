# import Activations
# import Losses
import numpy as np


# TODO: create kernel regularizer backward and forward

class Input:
    def __init__(self, size):
        self.output_shape = size

    def forward(self, input):
        return input

    def backward(self, input):
        pass


class Dense:
    def __init__(self, units, activation=None, use_bias=True, kernel_regularizer=None):
        self.output_shape = units
        self.activation = activation
        self.use_bias = use_bias
        self.bias = np.zeros((units, 1))
        self.kernel_regularizer = kernel_regularizer
        self.input_size = None
        self.prev_layer_out = None

    def __call__(self, inputs):
        """

        :param inputs:
        :return:
        """

        self.prev_layer = inputs
        self.input_size = self.prev_layer.output_shape
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(self.input_size, self.output_shape))
        return self

    def forward(self, input):
        """

        :param input:
        :return:
        """

        self.prev_layer_out = self.prev_layer.forward(input)
        weighted_sum = np.transpose(self.weights) @ self.prev_layer_out.transpose()
        if self.use_bias:
            bias = np.repeat(self.bias, weighted_sum.shape[-1], axis=-1)
            weighted_sum += bias
        if self.activation is None:
            return weighted_sum.transpose()
        return self.activation.forward(weighted_sum.transpose())

    def backward(self, temp_gradient):
        """

        :param temp_gradient:
        :return:
        """

        temp_gradient = self.activation.backward(temp_gradient) if self.activation is not None else temp_gradient
        next_grad = np.transpose(self.weights @ np.transpose(temp_gradient))

        # Calculate gradients for weights and bias
        delta_weights = self.prev_layer_out.transpose() @ temp_gradient
        delta_bias = temp_gradient.transpose().sum(axis=-1, keepdims=True)

        # Update weights
        self.weights -= 0.001 * delta_weights
        # print("gradients", delta_weights.flatten().mean())
        self.bias -= 0.001 * delta_bias

        self.prev_layer.backward(next_grad)


if __name__ == '__main__':
    inp = Input(1)
    b = Dense(40, activation=Activations.ReLu(), use_bias=True)(inp)
    # b = Dense(50, activation=Activations.ReLu())(b)
    # b = Dense(60, activation=Activations.ReLu())(b)
    # b = Dense(100, activation=Activations.ReLu())(inp)
    # b = Dense(10, activation=Activations.ReLu())(b)
    # b = Dense(2, activation=Activations.Linear())(b)
    b = Dense(4, activation=Activations.Softmax(), use_bias=True)(b)
    # b = Activations.ReLu()(b)
    # b = Activations.Linear()(b)
    # b = Activations.Linear()(b)
    # b = Activations.Linear()(b)
    # b = Activations.Tanh()(b)
    # b = Activations.Tanh()(b)
    # b = Activations.Tanh()(b)
    # b = Dense(2, activation=Activations.Linear())(b)
    for x in range(1000):
        outp = b.forward(np.array([[1], [-1]]))
        print(outp, outp.shape)
        # loss = Losses.L2()
        loss = Losses.Cross_Entropy()
        print("loss", loss.forward(outp, np.array([[0.4, 0.1, 0.1, 0.4], [0.4, 0.2, 0.2, 0.2]])))
        print("loss", loss.backward(outp, np.array([[0.4, 0.1, 0.1, 0.4], [0.4, 0.2, 0.2, 0.2]])))
        b.backward(loss.backward(outp, np.array([[0.4, 0.1, 0.1, 0.4], [0.4, 0.2, 0.2, 0.2]])))
        # b.backward(loss.backward(outp, np.array([[1], [0]])))
        # print("loss", loss.forward(outp, np.array([[1], [0]])))
        # b.backward(loss.backward(outp, np.array([[0.1, 0.9], [0.9, 0.1]])))
        # print("loss", loss.forward(outp, np.array([[0.1, 0.9], [0.9, 0.1]])))
