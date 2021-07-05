# main.py -- Entry point for training/testing LeNet-5 model

import os
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor

# Custom modules -- the model class and the training/testing functions
from lenet5 import LeNet5
from training import train
from testing import test


###############################################
# Model input and output shapes
# Since we're using MNIST, the images
# are 32x32, and there are 10 classes
INPUT_DIM = 28
OUTPUT_DIM = 10

# The dataset related parameters
# If the dataset was already downloaded, set
# NEED_DOWNLOAD to False
DATASET_DIR = './'
NEED_DOWNLOAD = True

BATCH_SIZE = 32
###############################################


def prepare_dataset():
    """ Loads the dataset (downloads if necessary) and returns data-loaders of training and testing sets """
    transform_fn = ToTensor()

    train_data = MNIST(root=DATASET_DIR, train=True, download=NEED_DOWNLOAD, transform=transform_fn)
    test_data = MNIST(root=DATASET_DIR, train=False, download=NEED_DOWNLOAD, transform=transform_fn)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


if __name__ == '__main__':

    train_loader, test_loader = prepare_dataset()

    # NOTE: Torch represents images as (Channels, Height, Width)
    #       Each sample in the train/test loaders are hence (batch_size, 1, 28, 28)
    # Create the model and train it
    model = LeNet5(INPUT_DIM, OUTPUT_DIM)

    train(train_loader, model)
    test(test_loader, model)