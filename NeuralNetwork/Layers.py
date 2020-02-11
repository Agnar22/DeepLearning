# import Activations
# import Losses
import numpy as np


class Input:
    num_input_layers = 0

    def __init__(self, size, name=None):
        self.output_shape = size
        self.name = name if name is not None else Input._set_default_name()

    @staticmethod
    def _set_default_name() -> str:
        layer_name = "Input_{0:d}".format(Input.num_input_layers)
        Input.num_input_layers += 1
        return layer_name

    def forward(self, input: np.ndarray) -> np.ndarray:
        return input

    def backward(self, input: np.ndarray) -> int:
        return 0


class Dense:
    num_dense_layers = 0

    def __init__(self, units, activation=None, use_bias=True, regularizer=None, name=None):
        self.output_shape = units
        self.activation = activation
        self.use_bias = use_bias
        self.bias = np.zeros((units, 1))
        self.regularizer = regularizer
        self.name = name if name is not None else Dense._set_default_name()
        self.input_size = None
        self.prev_layer_out = None
        self.lr = None

    @staticmethod
    def _set_default_name() -> str:
        """

        :return:
        """

        layer_name = "Dense_{0:d}".format(Dense.num_dense_layers)
        Dense.num_dense_layers += 1
        return layer_name

    def __call__(self, inputs: any) -> 'Dense':
        """

        :param inputs:
        :return:
        """

        self.prev_layer = inputs
        self.input_size = self.prev_layer.output_shape
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(self.input_size, self.output_shape))
        return self

    def __str__(self) -> str:
        """

        :return:
        """

        return 'Weights:\n{0} \n\n Bias:\n{1}'.format(str(list(self.weights)), str(list(self.bias)))

    def set_lr(self, lr: float) -> None:
        """
        Setting the learning rate for the layer
        :arg
        """
        self.lr = lr

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Propagates the input through the layer by calling the previous layer
        passing the input through the weights, adding bias and applying the
        activation function.
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

    def backward(self, temp_gradient: np.ndarray) -> np.float64:
        """
        Calculating the gradient for the weight and bias matrix and adds the gradients.
        Sending the gradients to the previous layer.
        :param temp_gradient:
        :return:
        """

        temp_gradient = self.activation.backward(temp_gradient) if self.activation is not None else temp_gradient
        next_grad = np.transpose(self.weights @ np.transpose(temp_gradient))

        # Calculate gradients for weights and bias
        delta_weights = self.prev_layer_out.transpose() @ temp_gradient
        delta_bias = temp_gradient.transpose().sum(axis=-1, keepdims=True)

        # Updating weights and bias according to reg_loss
        reg_loss = 0
        if self.regularizer is not None:
            reg_loss = self.regularizer.loss(self.weights) + self.regularizer.loss(self.bias)
            self.weights -= self.lr * self.regularizer.regularizer(self.weights)
            self.bias -= self.lr * self.regularizer.regularizer(self.bias)

        # Update weights and bias
        self.weights -= self.lr * delta_weights
        self.bias -= self.lr * delta_bias

        return reg_loss + self.prev_layer.backward(next_grad)

    def store_as_txt(self, path: str) -> None:
        """
        Storing the weights and bias as txt file
        :param path:
        :return:
        """

        with open(path + '/' + self.name + '.txt', "w+") as f:
            f.write(str(self))

    def save_weights(self, path: str) -> str:
        """

        :param path:
        :return:
        """

        print("Saving weights", path, self.name)

        file_path = path + '/' + self.name
        np.save(file_path, self.weights)
        return file_path

    def load_weights(self, file_path: str) -> None:
        print("Loaded", file_path + '.npy')
        self.weights = np.load(file_path + '.npy')


if __name__ == '__main__':
    inp = Input(1)
    b = Dense(100, activation=Activations.ReLu(), use_bias=True)(inp)
    b = Dense(100, activation=Activations.ReLu(), use_bias=True)(b)
    b = Dense(100, activation=Activations.ReLu(), use_bias=True)(b)
    b = Dense(4, activation=Activations.Softmax())(b)
    for x in range(1000):
        outp = b.forward(np.array([[1], [0.1]]))
        print(outp, outp.shape)
        # loss = Losses.L2()
        loss = Losses.Cross_Entropy()
        print("loss", loss.forward(outp, np.array([[0.3, 0.2, 0.3, 0.2], [0.4, 0.1, 0.1, 0.4]])))
        print("loss", loss.backward(outp, np.array([[0.3, 0.2, 0.3, 0.2], [0.4, 0.1, 0.1, 0.4]])))
        b.backward(loss.backward(outp, np.array([[0.3, 0.2, 0.3, 0.2], [0.4, 0.1, 0.1, 0.4]])))
        # b.backward(loss.backward(outp, np.array([[1], [0]])))
        # print("loss", loss.forward(outp, np.array([[1], [0]])))
        # b.backward(loss.backward(outp, np.array([[0.1, 0.9], [0.9, 0.1]])))
        # print("loss", loss.forward(outp, np.array([[0.1, 0.9], [0.9, 0.1]])))
