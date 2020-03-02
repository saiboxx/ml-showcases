import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset
from utils.model import NeuralNetwork
from utils.layer import Dense
from utils.activations import Sigmoid, Tanh, Relu
from utils.losses import MSE
from utils.optimizer import SGD
from sklearn.datasets import make_moons

EPOCHS = 100
BATCH_SIZE = 32


def main():
    # Load data
    # data = Dataset()
    # data = np.random.randn(1, 5)
    # label = np.random.randn(1)
    x, y = make_moons(n_samples=1000, random_state=42, noise=0.01)

    # Build model architecture
    model = NeuralNetwork()
    model.add(Dense(2, 50))
    model.add(Relu())
    model.add(Dense(50, 20))
    model.add(Relu())
    model.add(Dense(20, 1))
    model.add(Sigmoid())
    model.add_loss(MSE())
    optimizer = SGD()
    print(model)

    print('Start training with {} epochs'.format(EPOCHS))
    losses = []
    accuracies = []
    for e in range(EPOCHS):
        # Generated shuffled index vector
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        indices = np.array_split(indices, len(indices) // BATCH_SIZE)

        for batch_indices in indices:

            for i in batch_indices:
                x_batch = x[i, :]
                x_batch = x_batch[:, np.newaxis].transpose()

                # Forward propagation
                y_hat = model.forward(x_batch)

                # Get error
                model.get_loss(y[i], y_hat)

                # Backpropagation of error
                model.backward()
            optimizer.step(model)
            model.zero_grads()

        # Check training error
        y_hat = model.forward(x)
        loss = model.get_loss(y, y_hat)
        losses.append(loss)

        y_hat_class = np.where(y_hat.squeeze() < 0.5, 0, 1)
        accuracy = 1 - (sum(abs(y - y_hat_class)) / len(y_hat_class))
        accuracies.append(accuracy)
        print('Accuracy: {}'.format(accuracy))

    # Show training process
    plt.plot(accuracies)
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
