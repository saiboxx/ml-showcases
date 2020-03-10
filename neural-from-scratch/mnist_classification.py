import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
from dataset import Dataset
from utils.model import NeuralNetwork
from utils.layer import Dense
from utils.activations import Sigmoid, Tanh, Relu, LeakyRelu, ELU, Softmax
from utils.losses import MSE, CrossEntropy
from utils.optimizer import SGD, Adam

###############################################################################
# SET PARAMETERS                                                              #
###############################################################################
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
###############################################################################


def main():

    # Load MNIST
    data = Dataset()

    # Reduce data for testing
    #data.train = data.train[:3000, :]
    #data.train_label = data.train_label[:3000]

    ###############################################################################
    # BUILD MODEL ARCHITECTURE                                                    #
    ###############################################################################
    model = NeuralNetwork()
    model.add(Dense(784, 256, "he"))
    model.add(ELU())
    model.add(Dense(256, 128, "he"))
    model.add(ELU())
    model.add(Dense(128, 64, "he"))
    model.add(ELU())
    model.add(Dense(64, 10, "he"))
    model.add(Softmax())
    model.add_loss(CrossEntropy())
    optimizer = Adam(alpha=LEARNING_RATE)
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
    fig, ax1 = plt.subplots()
    ax1.set_title('MNIST Training Progress')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    lns1 = ax1.plot(train_accuracies, label='Training Accuracy', color='#1f77b4')
    lns2 = ax1.plot(test_accuracies, label='Test Accuracy', color='#17becf')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Error')
    lns3 = ax2.plot(train_losses, label='Training Loss', color='#ff7f0e')
    lns4 = ax2.plot(test_losses, label='Test Loss', color='#d62728')
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
