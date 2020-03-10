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
    def __init__(self, alpha: Optional[float] = 0.001):
        super(SGD, self).__init__(alpha)

    def step(self, model: NeuralNetwork):
        for module in model.modules:
            if hasattr(module, 'gradients') and module.gradients is not None:
                gradients = []
                for k, v in module.gradients.items():
                    gradients.append(sum(v) / len(v))

                update_step = [self.alpha * grad for grad in gradients]
                module.update(update_step)


class Adam(Optimizer):
    def __init__(self,
                 alpha: Optional[float] = 0.001,
                 beta1: Optional[float] = 0.9,
                 beta2: Optional[float] = 0.999,
                 epsilon: Optional[float] = 1e-8, ):
        super(Adam, self).__init__(alpha)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.first_step = True
        self.first_mom = []
        self.second_mom = []

    def step(self, model: NeuralNetwork):
        # Initialize gradient structure with zeros
        if self.first_step:
            self.start_up(model)

        i = 0
        for module in model.modules:
            if hasattr(module, 'gradients') and module.gradients is not None:
                update_step = []
                for j, (k, v) in enumerate(module.gradients.items()):
                    # Get averaged gradient
                    gradient = sum(v) / len(v)
                    # Update first moment estimate
                    self.first_mom[i][j] = self.beta1 * self.first_mom[i][j] + (1. - self.beta1) * gradient
                    # Update second moment estimate
                    self.second_mom[i][j] = self.second_mom[i][j] + (1. - self.beta2) * np.square(gradient)
                    # Correct bias in first moment
                    #self.first_mom[i][j] = self.first_mom[i][j] / (1. - self.beta1)
                    # Correct bias in second moment
                    #self.second_mom[i][j] = self.second_mom[i][j] / (1. - self.beta2)
                    # Compute update
                    update = self.alpha * self.first_mom[i][j] / (np.sqrt(self.second_mom[i][j]) + self.epsilon)
                    update_step.append(update.astype(np.float64))

                module.update(update_step)
                i += 1

    def start_up(self, model: NeuralNetwork):
        self.first_step = False
        for module in model.modules:
            if hasattr(module, 'gradients') and module.gradients is not None:
                gradient_structure = []
                for k, v in module.gradients.items():
                    gradient_structure.append(np.zeros_like(v[0], dtype=np.float64))

                self.first_mom.append(gradient_structure[:])
                self.second_mom.append(gradient_structure[:])
