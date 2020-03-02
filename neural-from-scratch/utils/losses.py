import numpy as np


class Loss(object):
    def __init__(self):
        self.last_y = None
        self.last_y_hat = None

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        pass

    def backward(self):
        pass


class MSE(Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        self.last_y = y
        self.last_y_hat = y_hat
        return 0.5 * np.square(np.subtract(y, y_hat)).mean()

    def backward(self) -> np.ndarray:
        return - np.subtract(self.last_y, self.last_y_hat)
