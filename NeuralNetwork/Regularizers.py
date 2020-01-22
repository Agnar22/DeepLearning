import numpy as np


class Regularizer:
    def __init__(self, reg_func, reg_func_der):
        self.reg_func = reg_func
        self.reg_func_der = reg_func_der

    def loss(self, weights):
        return self.reg_func(weights)

    def regularizer(self, weights):
        return self.reg_func_der(weights)


class L2(Regularizer):
    def __init__(self):
        Regularizer.__init__(self, lambda x: (x * x).sum() / np.size, lambda x: x)
