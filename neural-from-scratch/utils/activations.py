import numpy as np
from typing import Optional


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


class LeakyRelu(Activation):
    def __init__(self, alpha: Optional[float] = 0.01):
        super(LeakyRelu, self).__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return np.where(x < 0, self.alpha * x,  x)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        x = np.ones_like(self.last_x)
        x[self.last_x < 0] = self.alpha
        return np.multiply(x, delta)


class ELU(Activation):
    def __init__(self, alpha: Optional[float] = 0.01):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return np.where(x < 0, self.alpha * (np.exp(x) - 1),  x)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        x = np.where(self.last_x < 0, self.alpha * np.exp(self.last_x),  1)
        return np.multiply(x, delta)


class Softmax(Activation):
    def __init__(self,):
        super(Softmax, self).__init__()
        self.last_activation = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        self.last_activation = e_x / np.sum(e_x, axis=1)[:, None]
        return self.last_activation

    def backward(self, delta: np.ndarray, ce: Optional[bool] = True) -> np.ndarray:
        # The derivative is already calculated when using Cross Entropy Loss.
        if ce:
            return delta
        else:
            # Create Jacobi matrix
            # Off diagonal
            jacobi = - np.multiply(self.last_activation[:, :, None], self.last_activation[:, None, :])
            # Diagonal
            iy, ix = np.diag_indices_from(jacobi[0])
            jacobi[:, iy, ix] = np.multiply(self.last_activation, (1 - self.last_activation))
            # Collapse
            return np.multiply(np.sum(jacobi, axis=1), delta)