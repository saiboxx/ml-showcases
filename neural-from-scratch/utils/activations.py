import numpy as np


class Activation(object):
    def __init__(self):
        self.last_x = None

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def forward(self, x: np.ndarray):
        pass

    def backward(self, delta: np.ndarray):
        pass


class Identity(Activation):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return x

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return np.multiply(np.ones(self.last_x), delta)


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return np.multiply(self.forward(self.last_x) * (1 - self.forward(self.last_x)), delta)


class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return np.tanh(x)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return np.multiply(1 - np.square(np.tanh(self.last_x)), delta)


class Relu(Activation):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return x * (x > 0)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return np.multiply(1 * (self.last_x > 0), delta)
