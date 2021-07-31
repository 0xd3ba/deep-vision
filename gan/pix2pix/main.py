# main.py -- Entry point for training/testing Pix2Pix model


import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor

# Custom modules -- the dataset class, model class and the training/testing functions
from dataset.danbooru_sketch_pair import DanbooruSketchPair
from modules.generator import Generator
from modules.discriminator import Discriminator
from training import train


################################################
# The dataset related parameters
DATASET_DIR = './dataset/danbooru_sketch_pair/'
DATASET_SKETCH_DIR = 'sketch'
DATASET_SRC_DIR = 'src'

# WARNING: The dataset I'm using only has
# SIX images. Don't exceed that value.
BATCH_SIZE = 6
################################################


def prepare_loader():
    """ Loads the dataset (downloads if necessary) and returns data-loaders of training and testing sets """

    dataset = DanbooruSketchPair(DATASET_DIR, DATASET_SKETCH_DIR, DATASET_SRC_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader


if __name__ == '__main__':

    # NOTE: Torch represents images as (Channels, Height, Width)
    #       Each sample in the train/test loaders are hence (batch_size, 3, 224, 224)
    # Create the model and train it

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = prepare_loader()
    generator_model = Generator()
    discriminator_model = Discriminator()

    train(data_loader, generator_model, discriminator_model, device)
