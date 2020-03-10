import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
from dataset import Dataset
from utils.model import NeuralNetwork
from utils.layer import Dense
from utils.activations import Sigmoid, Tanh, Relu, LeakyRelu, ELU, Softmax
from utils.losses import MSE, CrossEntropy
from utils.optimizer import SGD

###############################################################################
# SET PARAMETERS                                                              #
###############################################################################
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_SAMPLES = 250
###############################################################################


def main():

    # Load MNIST
    data = Dataset()

    # Reduce data for testing
    #data.train = data.train[:300, :]
    #data.train_label = data.train_label[:300]

    ###############################################################################
    # BUILD MODEL ARCHITECTURE                                                    #
    ###############################################################################
    model = NeuralNetwork()
    model.add(Dense(784, 128, "he"))
    model.add(ELU())
    model.add(Dense(128, 32, "he"))
    model.add(ELU())
    model.add(Dense(32, 10, "he"))
    model.add(Softmax())
    model.add_loss(CrossEntropy())
    optimizer = SGD(alpha=LEARNING_RATE)
    ###############################################################################

    print(model)

    print('Start training with {} epochs'.format(EPOCHS))
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for e in range(EPOCHS):
        # Generated shuffled index vector
        indices = np.arange(data.train.shape[0])
        np.random.shuffle(indices)
        indices = np.array_split(indices, len(indices) // BATCH_SIZE)

        for batch_indices in tqdm(indices):
            for i in batch_indices:
                x_batch = data.train[i, :]
                x_batch = x_batch[:, np.newaxis].transpose()

                # Forward propagation
                y_hat = model.forward(x_batch)

                # Get error
                model.get_loss(data.train_label[i], y_hat)

                # Backpropagation of error
                model.backward()

            # Apply optimizer and reset gradients
            optimizer.step(model)
            model.zero_grads()

        # Check training error
        y_hat = model.forward(data.train)
        train_loss = model.get_loss(data.train_label, y_hat)
        train_losses.append(train_loss)
        y_hat_class = np.argmax(y_hat, axis=1)
        y_class = np.argmax(data.train_label, axis=1)
        train_accuracy = np.sum(np.where(y_class == y_hat_class, 1, 0)) / len(y_hat_class)
        train_accuracies.append(train_accuracy)

        # Check validation error
        y_hat = model.forward(data.test)
        test_loss = model.get_loss(data.test_label, y_hat)
        test_losses.append(test_loss)
        y_hat_class = np.argmax(y_hat, axis=1)
        y_class = np.argmax(data.test_label, axis=1)
        test_accuracy = np.sum(np.where(y_class == y_hat_class, 1, 0)) / len(y_hat_class)
        test_accuracies.append(test_accuracy)
        print('Ep. {0:>4} Accuracy: {1:3.3f} Loss: {2:3.3f} Val-Accuracy: {3:3.3f} Val-Loss: {4:3.3f}'
              .format(e, train_accuracy, train_loss, test_accuracy, test_loss))

    # Plot results
    plot_results(train_accuracies, train_losses, test_accuracies, test_losses)


def plot_results(train_accuracies: list, train_losses: list, test_accuracies: list, test_losses: list):
    plt.title('Training Progress')
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
