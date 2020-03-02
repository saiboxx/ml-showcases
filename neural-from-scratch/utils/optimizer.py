import numpy as np
from utils.model import NeuralNetwork
from typing import Optional


class Optimizer(object):
    def __init__(self, alpha: Optional[float] = 0.001):
        self.alpha = alpha

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def step(self, model: NeuralNetwork):
        pass


class SGD(Optimizer):
    def __init__(self):
        super(SGD, self).__init__()

    def step(self, model: NeuralNetwork):
        for module in model.modules:
            if hasattr(module, 'gradients') and len(module.gradients) > 0:
                update_step = [self.alpha * grad for grad in module.gradients]
                module.update(update_step)
