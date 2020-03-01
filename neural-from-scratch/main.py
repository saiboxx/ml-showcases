import numpy as np
from dataset import Dataset


def main():
    data = Dataset()
    print(np.max(data.test))


if __name__ == '__main__':
    main()
