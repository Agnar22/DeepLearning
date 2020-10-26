# import Layers
import numpy as np


class Sequential:
    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.lr = None

    def add(self, layer: any) -> None:
        """
        Adding a layer to the neural network
        :param layer:
        :return:
        """

        if len(self.layers) > 0:
            layer(self.layers[-1])
        self.layers.append(layer)

    def save_model(self, file_path: str, as_txt: [bool] = False) -> list:
        """

        :param file_path:
        :return:
        """
        if as_txt:
            files = list(filter(lambda x: x is not None,
                                map(lambda x: x.store_as_txt(file_path) if hasattr(x, 'weights') else None,
                                    self.layers)))
        else:
            files = list(filter(lambda x: x is not None,
                                map(lambda x: x.save_weights(file_path) if hasattr(x, 'weights') else None,
                                    self.layers)))
        return files

    def load_model(self, file_paths: list) -> None:
        """

        :param file_paths:
        :return:
        """

        for file_path in file_paths:
            layer_name = file_path.split('/')[-1]
            list(filter(lambda x: x.name == layer_name, self.layers))[0].load_weights(file_path)

    def compile(self, loss: float = None, lr: float = None) -> None:
        """

        :param loss:
        :param lr:
        :return:
        """

        self.loss_func = loss
        self.lr = lr

        added_layers = []
        for layer in self.layers:
            if not hasattr(layer, 'prev_layer'):
                continue
            curr_ancestor = layer
            while hasattr(curr_ancestor, 'prev_layer'):
                curr_ancestor.set_lr(self.lr)
                curr_ancestor = curr_ancestor.prev_layer
                if curr_ancestor not in self.layers: added_layers.append(curr_ancestor)
        added_layers.extend(self.layers)
        self.layers = added_layers

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, validation_data: tuple = None,
            epochs: int = 10, batch_size: int = 64) -> tuple:
        """
        Training the current network on x_train with targets y_train.
        :param x_train:
        :param y_train:
        :param validation_data: (x_val, y_val)
        :param epochs:
        :param batch_size:
        :return:
        """

        train_loss = []
        val_loss = []
        reg_loss = []
        for epoch in range(epochs):
            correct = 0
            loss = 0
            curr_reg_loss = 0
            for x in range(0, x_train.shape[0], batch_size):
                # Setting up current batch
                increment = min(batch_size, x_train.shape[0] - x)
                mini_x_train = x_train[x:x + increment].reshape(increment, x_train.shape[-1])
                mini_y_train = y_train[x:x + increment]

                # Forward propagation - prediction
                fwd_propagation, temp_correct, temp_loss = self.predict(mini_x_train, mini_y_train)
                correct += temp_correct
                loss += temp_loss * increment

                # Back-propagation
                loss_grad = self.loss_func.backward(fwd_propagation, mini_y_train)
                curr_reg_loss += self.layers[-1].backward(loss_grad) * increment

                Sequential.print_progress(40, x + increment, x_train.shape[0], loss, correct, curr_reg_loss/(x+increment))
            train_loss.append(loss / x_train.shape[0])
            reg_loss.append(curr_reg_loss / x_train.shape[0])

            if validation_data is not None:
                x_val, y_val = validation_data

                # Forward propagation - prediction
                fwd_propagation, correct_val, curr_loss = self.predict(x_val, y_val)

                Sequential.print_progress(40, x_train.shape[0], x_train.shape[0], loss, correct, reg_loss[-1], val_loss=curr_loss,
                                          correct_val=correct_val, num_val=y_val.shape[0])
                val_loss.append(curr_loss)
            print()
        return train_loss, val_loss

    def predict(self, x: np.ndarray, y: np.ndarray = None) -> tuple:
        """
        Propagating x through the network and calculates loss if y not None
        :param x:
        :param y:
        :return:
        """

        fwd_propagation = self.layers[-1].forward(x)
        if y is not None:
            correct = (np.argmax(fwd_propagation, axis=-1) == np.argmax(y, axis=-1)).sum()
            loss = self.loss_func.forward(fwd_propagation, y)
            return fwd_propagation, correct, loss
        return fwd_propagation, None, None

    @staticmethod
    def print_progress(bars, batch_end, epoch_length, sum_loss, correct, reg_loss, val_loss=None, correct_val=None,
                       num_val=None) -> None:
        """
        Printing the progressbar for an epoch
        :param bars:
        :param batch_end:
        :param epoch_length:
        :param sum_loss:
        :param correct:
        :param val_loss:
        :param correct_val:
        :param num_val:
        :return:
        """

        r = int((batch_end / epoch_length) * bars)
        progressbar = '\r[' + ''.join('=' for _ in range(r)) + '>' + ''.join('-' for _ in range(bars - r)) + '] '
        progress = '{0:.2f} % ({1:d}/{2:d})'.format(batch_end * 100 / epoch_length, batch_end, epoch_length)
        train_stats = '\t\tloss: {0:.7f}  corr: {1:d}/{2:d} ({3:.2f} %) reg_loss: {4:.5f}'.format(sum_loss / batch_end, correct,
                                                                                batch_end,
                                                                                100 * correct / batch_end, reg_loss)

        val_stats = ''
        if val_loss is not None:
            val_stats = '\t\tval_loss: {0:.7f} val_corr: {1:d}/{2:d} ({3:.2f} %)'.format(val_loss, correct_val,
                                                                                         num_val,
                                                                                         100 * correct_val / num_val)

        print(progressbar + progress + train_stats + val_stats, end='')


if __name__ == '__main__':
    Sequential().add(Layers.Input(3))
