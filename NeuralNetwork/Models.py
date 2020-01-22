# import Layers
import numpy as np


class Sequential:
    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.lr = None

    def add(self, layer):
        if len(self.layers) > 0:
            layer(self.layers[-1])
        self.layers.append(layer)

    def save_weights(self, file_path):
        pass

    def load_weights(self, file_path):
        pass

    def compile(self, loss=None, lr=None):
        self.loss_func = loss
        self.lr = lr

    def fit(self, x_train, y_train, validation_data=None, epochs=10, batch_size=32):
        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            correct = 0
            loss = 0
            for x in range(0, x_train.shape[0], batch_size):
                increment = min(batch_size, x_train.shape[0] - x)

                fwd_propagation = self.layers[-1].forward(
                    x_train[x:x + increment].reshape(increment, x_train.shape[-1]))
                correct += (np.argmax(fwd_propagation, axis=-1) == np.argmax(y_train[x:x + increment], axis=-1)).sum()
                temp_loss = self.loss_func.forward(fwd_propagation, y_train[x:x + increment])
                loss += temp_loss * increment

                loss_grad = self.loss_func.backward(fwd_propagation, y_train[x:x + increment])
                self.layers[-1].backward(loss_grad)

                Sequential.print_progress(40, x + increment, x_train.shape[0], loss, correct)
            train_loss.append(loss / x_train.shape[0])

            if validation_data is not None:
                x_val, y_val = validation_data

                fwd_propagation = self.layers[-1].forward(x_val)
                correct_val = (np.argmax(fwd_propagation, axis=-1) == np.argmax(y_val, axis=-1)).sum()
                curr_loss = self.loss_func.forward(fwd_propagation, y_val)

                Sequential.print_progress(40, x_train.shape[0], x_train.shape[0], loss, correct, loss_val=curr_loss, correct_val=correct_val, val_samp=y_val.shape[0])
            val_loss.append(curr_loss)
            print()
        return train_loss, val_loss

    def predict(self, x, y=None):
        pass

    @staticmethod
    def print_progress(bars, batch_end, epoch_length, sum_loss, correct,
                       loss_val=None, correct_val=None, val_samp=None):
        r = int((batch_end / epoch_length) * bars)
        progressbar = '\r[' + ''.join('=' for _ in range(r)) + '>' + ''.join('-' for _ in range(bars - r)) + '] '
        progress = str(round(batch_end * 100 / epoch_length, 3)) + ' % (' + \
                   str(batch_end) + '/' + str(epoch_length) + ')'
        train_stats = '\tloss: ' + str(round(sum_loss / batch_end, 5)) + \
                      '\tcor: ' + str(correct) + '/' + str(batch_end) + \
                      ' (' + str(round(correct * 100 / batch_end, 4)) + ' %)'

        val_stats = ''
        if loss_val is not None:
            val_stats = '\tval_loss: ' + str(round(loss_val, 5)) + \
                        '\tval_cor: ' + str(correct_val) + '/' + str(val_samp) + \
                        ' (' + str(round(correct_val * 100 / val_samp, 4)) + ' %)'
        print(progressbar + progress + train_stats + val_stats, end='')


if __name__ == '__main__':
    Sequential().add(Layers.Input(3))
