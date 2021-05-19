from visualisation.dependencies.load_dataset import load_dataset

import sys


def main():
    dataset = load_dataset(sys.argv[1])

    print(dataset)

    return None


if __name__ == '__main__':
    main()
