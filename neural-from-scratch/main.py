import numpy as np
from dataset import Dataset
from utils.model import NeuralNetwork
from utils.layer import Dense
from utils.activations import Sigmoid, Tanh, Relu
from utils.losses import MSE
from utils.optimizer import SGD
from sklearn.datasets import make_moons

EPOCHS = 1000
BATCH_SIZE = 1


def main():
    # Load data
    #data = Dataset()
    #data = np.random.randn(1, 5)
    #label = np.random.randn(1)
    X, y = make_moons(n_samples=200, random_state=42, noise=0.1)

    # Build model architecture
    model = NeuralNetwork()
    model.add(Dense(2, 10))
    model.add(Relu())
    model.add(Dense(10, 20))
    model.add(Relu())
    model.add(Dense(20, 1))
    model.add(Sigmoid())
    model.add_loss(MSE())
    optimizer = SGD()
    print(model)

    print('Start training with {} epochs'.format(EPOCHS))
    losses = []
    for e in range(EPOCHS):
        for i in range(len(X)):
            data = X[i, :]
            data = data[:, np.newaxis].transpose()
            label = y[i]

            # Forward propagation
            y_hat = model.forward(data)

            # Get error
            model.get_loss(label, y_hat)

            # Backpropagation of error
            model.backward()
            optimizer.step(model)

        # Check training error
        y_hat = model.forward(X)
        loss = model.get_loss(y, y_hat)
        losses.append(loss)

        y_hat_class = np.where(y_hat.squeeze() < 0.5, 0, 1)
        accuracy = 1 - (sum(abs(y - y_hat_class)) / len(y_hat_class))
        print('Accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
