import numpy as np
from functools import reduce
from utils.losses import Loss


class NeuralNetwork(object):
    def __init__(self):
        self.modules = []
        self.loss = None

    def __str__(self):
        str_list = ['Neural Network Architecture:\n']
        [str_list.append(str(i) + '\t' + repr(m) + '\n')
         for i, m in enumerate(self.modules, start=1)]

        str_list.append("Loss:\t{}".format(self.loss))
        return ''.join(str_list)

    def add(self, module: object):
        self.modules.append(module)

    def add_loss(self, loss: Loss):
        self.loss = loss

    def forward(self, x: np.ndarray) -> np.ndarray:
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self):
        error = self.loss.backward()
        for module in reversed(self.modules):
            error = module.backward(error)

    def get_loss(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return self.loss.forward(y, y_hat)

    def zero_grads(self):
        for module in self.modules:
            if hasattr(module, 'gradients'):
                module.gradients = []
