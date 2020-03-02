import numpy as np
from typing import Optional


class Layer(object):
    def __init__(self, input: int, output: int, initializer: Optional[str]):
        self.input = input
        self.output = output
        self.initializer = initializer
        self.gradients = []
        self.last_x = None

    def __repr__(self):
        return '{}\tIn: {}\tOut: {}'\
            .format(self.__class__.__name__, self.input, self.output)

    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def backward(self, delta: np.ndarray) -> np.ndarray:
        pass

    def get_gradients(self, delta: np.ndarray):
        pass

    def update(self, gradients: np.ndarray):
        pass


class Dense(Layer):
    def __init__(self, input: int, output: int, initializer: Optional[str] = None):
        super(Dense, self).__init__(input, output, initializer)

        if initializer == 'he':
            self.weights = np.random.randn(input, output) * np.sqrt(2 / input)
        elif initializer == 'glorot':
            self.weights = np.random.uniform(low=-np.sqrt(6 / (input + output)),
                                             high=np.sqrt(6 / (input + output)),
                                             size=(input, output))
        else:
            self.weights = np.random.randn(input, output)

        self.bias = np.zeros(output)
        self.trainable_params = [self.weights, self.bias]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return x.dot(self.weights) + self.bias

    def backward(self, delta: np.ndarray) -> np.ndarray:
        self.get_gradients(delta)
        return delta.dot(self.weights.transpose())

    def get_gradients(self, delta: np.ndarray):
        grad_weights = self.last_x.transpose().dot(delta)
        grad_bias = delta
        self.gradients = [grad_weights, grad_bias]

    def update(self, gradients: list):
        self.weights -= gradients[0]
        self.bias -= gradients[1].squeeze()
