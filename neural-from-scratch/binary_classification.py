import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils.model import NeuralNetwork
from utils.layer import Dense
from utils.activations import Sigmoid, Tanh, Relu, LeakyRelu, ELU
from utils.losses import MSE, BinaryCrossEntropy
from utils.optimizer import SGD
from sklearn.datasets import make_moons, make_circles

###############################################################################
# SET PARAMETERS                                                              #
###############################################################################
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_SAMPLES = 250
###############################################################################

parser = ArgumentParser()
parser.add_argument('--moons', dest='moons', action='store_true')
args = parser.parse_args()


def main():

    # Generate training data
    if args.moons:
        x, y = make_moons(n_samples=NUM_SAMPLES, random_state=42, noise=0.1)
    else:
        x, y = make_circles(n_samples=NUM_SAMPLES, random_state=42, noise=0.1, factor=0.2)

    ###############################################################################
    # BUILD MODEL ARCHITECTURE                                                    #
    ###############################################################################
    model = NeuralNetwork()
    model.add(Dense(2, 64, "he"))
    model.add(ELU())
    model.add(Dense(64, 32, "he"))
    model.add(ELU())
    model.add(Dense(32, 1, "he"))
    model.add(Sigmoid())
    model.add_loss(BinaryCrossEntropy())
    optimizer = SGD(alpha=LEARNING_RATE)
    ###############################################################################

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

            # Apply optimizer and reset gradients
            optimizer.step(model)
            model.zero_grads()

        # Check training error
        y_hat = model.forward(x)
        loss = model.get_loss(y, y_hat)
        losses.append(loss)

        y_hat_class = np.where(y_hat.squeeze() < 0.5, 0, 1)
        accuracy = 1 - (sum(abs(y - y_hat_class)) / len(y_hat_class))
        accuracies.append(accuracy)
        print('Ep. {0:>4} Accuracy: {1:3.3f} Loss: {2:3.3f}'.format(e, accuracy, loss))

    # Plot results
    plot_results(model, x, y, accuracies, losses)


def plot_results(model: NeuralNetwork, data: np.ndarray, label: np.ndarray, accuracy: list, loss: list):
    x_min = min(data[:, 0]) - 0.2
    x_max = max(data[:, 0]) + 0.2
    y_min = min(data[:, 1]) - 0.2
    y_max = max(data[:, 1]) + 0.2

    xx, yy = np.mgrid[x_min:x_max:.05, y_min:y_max:.05]
    grid = np.c_[xx.ravel(), yy.ravel()]
    y_hat = model.forward(grid).reshape(xx.shape)
    y_hat_class = np.where(y_hat.squeeze() < 0.5, 0, 1)
    figure = plt.figure(figsize=(7, 7))
    plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    plt.title('Decision Boundary')
    plt.contourf(xx, yy, y_hat_class)
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.subplot2grid((3, 1), (2, 0))
    plt.title('Training Progress')
    plt.plot(accuracy, label='Accuracy')
    plt.plot(loss, label='Loss')
    plt.legend()
    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
