import numpy as np
import matplotlib.pyplot as plt
from utils.model import NeuralNetwork
from utils.layer import Dense
from utils.activations import Sigmoid, Tanh, Relu
from utils.losses import MSE
from utils.optimizer import SGD
from sklearn.datasets import make_moons

EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 0.1


def main():

    # Generate training data
    x, y = make_moons(n_samples=1000, random_state=42, noise=0.1)

    # Build model architecture
    model = NeuralNetwork()
    model.add(Dense(2, 64, "he"))
    model.add(Relu())
    model.add(Dense(64, 32, "he"))
    model.add(Relu())
    model.add(Dense(32, 1, "he"))
    model.add(Sigmoid())
    model.add_loss(MSE())
    optimizer = SGD(alpha=LEARNING_RATE)
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
        print('Ep. {0:>4} Accuracy: {1:3.3f} Loss: {2:3.3f}'.format(e, accuracy, loss))

    # Plot results
    plot_results(model, x, y, accuracies, losses)


def plot_results(model: NeuralNetwork, data: np.ndarray, label: np.ndarray, accuracy: list, loss: list):
    xx, yy = np.mgrid[-1.4:2.5:.05, -0.8:1.4:.05]
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
    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
