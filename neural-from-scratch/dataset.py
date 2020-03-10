import os
import pickle
import numpy as np
from typing import Tuple


class Dataset(object):
    """
    Data is available at
    https://www.kaggle.com/oddrationale/mnist-in-csv/data#mnist_test.csv
    """
    def __init__(self):
        self.train, self.train_label, self.test, self.test_label \
            = self.get_data()
        self.train /= 255
        self.test /= 255
        self.train_label = self.one_hot_encode(self.train_label)
        self.test_label = self.one_hot_encode(self.test_label)

    def get_data(self) -> Tuple:
        print('Loading training and validation data.')
        files = os.listdir('data')
        if "data.pkl" in files:
            train, test = pickle.load(open(os.path.join('data', 'data.pkl'), 'rb'))
        else:
            train = np.genfromtxt('data/mnist_train.csv', delimiter=',', skip_header=1)
            test = np.genfromtxt('data/mnist_test.csv', delimiter=',', skip_header=1)
            pickle.dump((train, test), open(os.path.join('data', 'data.pkl'), 'wb'))

        train_label = train[:, 0].astype(np.int)
        test_label = test[:, 0].astype(np.int)

        train = np.delete(train, axis=1, obj=0)
        test = np.delete(test, axis=1, obj=0)

        print('Training Data:\t{0} rows \t{1} columns'
              .format(train.shape[0], train.shape[1]))
        print('Test Data:\t{0} rows \t{1} columns'
              .format(test.shape[0], test.shape[1]))

        return train, train_label, test, test_label

    def one_hot_encode(self, label: np.ndarray) -> np.ndarray:
        num_classes = np.max(label) + 1
        return np.eye(num_classes)[label]
