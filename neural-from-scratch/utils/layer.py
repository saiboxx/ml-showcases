import numpy as np


class Dense(object):
    def __init__(self, input: int, output: int):
        self.size = {"input": input, "output": output}
        self.weights = np.random.randn(input, output)