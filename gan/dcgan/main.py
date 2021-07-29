# main.py -- Entry point for training/testing GAN model


import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor

# Custom modules -- the dataset class, model class and the training/testing functions
from dcgan import GeneratorCNN, DiscriminatorCNN
from dataset.animefaces import AnimeFaces
from training import train


################################################
# Model input and output shapes
# Each image is of shape (3, 64, 64)
INPUT_DIM = 64
NOISE_DIM = 10  # Dimension of the noise vector

# The dataset related parameters
DATASET_DIR = './dataset/animefaces/'

BATCH_SIZE = 32
################################################


def prepare_loader():
    """ Loads the dataset (downloads if necessary) and returns data-loaders of training and testing sets """

    dataset = AnimeFaces(DATASET_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader


if __name__ == '__main__':

    # NOTE: Torch represents images as (Channels, Height, Width)
    #       Each sample in the train/test loaders are hence (batch_size, 3, 224, 224)
    # Create the model and train it

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = prepare_loader()
    generator_model = GeneratorCNN(NOISE_DIM, INPUT_DIM)
    discriminator_model = DiscriminatorCNN()

    train(data_loader, generator_model, discriminator_model, device)
