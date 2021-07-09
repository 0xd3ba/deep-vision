# main.py -- Entry point for training/testing LeNet-5 model

import torch
from torch.utils.data.dataloader import DataLoader

# Custom modules -- the dataset class, model class and the training/testing functions
from dataset.imagenette import ImageNetteDataset
from vgg16 import VGG16
from training import train


###############################################
# The dataset related parameters. Extract the
# dataset and place it wherever you wish
# Just set the directory to it appropriately
DATASET_DIR = './dataset/imagenette2-320/'
ANNOTATION_CSV = 'noisy_imagenette.csv'

BATCH_SIZE = 32
###############################################


def prepare_dataset(train=True):
    """ Loads the dataset and return the data-loader of it """

    dataset = ImageNetteDataset(data_dir=DATASET_DIR, annotation_csv=ANNOTATION_CSV, train=train)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader


if __name__ == '__main__':

    train_loader = prepare_dataset(train=True)

    # NOTE: Torch represents images as (Channels, Height, Width)
    #       Each sample in the train/test loaders are hence (batch_size, 3, 224, 224)
    # Create the model and train it

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG16()

    train(train_loader, model, device)
