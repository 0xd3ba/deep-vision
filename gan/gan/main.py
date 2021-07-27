# main.py -- Entry point for training/testing GAN model


import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor

# Custom modules -- the dataset class, model class and the training/testing functions
from gan import Generator, Discriminator
from training import train


###############################################
# Model input and output shapes
# Since we're using MNIST, the images
# are 28x28, and there are 10 classes (which we
# have no need for)
INPUT_DIM = 28*28
NOISE_DIM = 10      # The dimension for noise

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
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader


if __name__ == '__main__':

    # NOTE: Torch represents images as (Channels, Height, Width)
    #       Each sample in the train/test loaders are hence (batch_size, 3, 224, 224)
    # Create the model and train it

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_class = prepare_dataset()
    generator_model = Generator(NOISE_DIM, INPUT_DIM)
    discriminator_model = Discriminator(INPUT_DIM)

    train(data_class, generator_model, discriminator_model, device)
