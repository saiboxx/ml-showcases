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


class CrossEntropy(Loss):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        y = np.reshape(y, y_hat.shape)
        self.last_y = y
        self.last_y_hat = y_hat
        return - np.sum(y * np.log(y_hat + 1e-9)) / y.shape[0]

    def backward(self) -> np.ndarray:
        # Backpropagation for use with Softmax!
        return np.sum(self.last_y_hat - self.last_y, axis=0)[None, :]


class BinaryCrossEntropy(Loss):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        self.last_y = y
        self.last_y_hat = y_hat
        if np.ndim(y) == 0:
            if y == 1:
                return -np.log(y_hat)
            else:
                return -np.log(1 - y_hat)
        else:
            y_hat = y_hat.squeeze()
            return np.sum(np.where(y == 1, - np.log(y_hat), - np.log(1 - y_hat))) / y.shape[0]

    def backward(self) -> np.ndarray:
        if self.last_y == 1:
            return -1 / self.last_y_hat
        else:
            return 1 / (1 - self.last_y_hat)
