import numpy as np


class Regularizer:
    def __init__(self, reg_func, reg_func_der, alpha=0):
        self.reg_func = reg_func
        self.reg_func_der = reg_func_der
        self.alpha = alpha

    def loss(self, weights):
        return self.alpha * self.reg_func(weights).sum() / weights.size

    def regularizer(self, weights):
        return self.alpha * self.reg_func_der(weights)


class L2(Regularizer):
    def __init__(self, alpha=0):
        Regularizer.__init__(self, lambda x: x * x, lambda x: x, alpha=alpha)


class L1(Regularizer):
    def __init__(self, alpha=0):
        Regularizer.__init__(self, lambda x: x, lambda x: np.ones(x.shape), alpha=alpha)


if __name__ == '__main__':
    print(L2().loss(np.array([[1, 2], [2, 3]])))
    print(L2().regularizer(np.array([[1, 2], [2, 3]])))
    print(L1().loss(np.array([[1, 2], [2, 3]])))
    print(L1().regularizer(np.array([[1, 2], [2, 3]])))
