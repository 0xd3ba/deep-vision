# main.py -- Entry point for training/testing FCN model
# NOTE: Dataloader can't be used here as it expects all inputs to be of same dimension
#       but in this case, the inputs are of variable width, so it's not possible to use it here.

import torch

# Custom modules -- the dataset class, model class and the training/testing functions
from dataset.voc2012 import VOC2012
from fcn import FCN
from training import train


###############################################
# WARNING: Don't change these parameters
# I'm using my own custom dataset and custom
# annotation file. The dataset is an extremely
# small subset of original Pascal VOC-2012
DATASET_DIR = './dataset/voc2012/'
ANNOTATION_CSV = 'annotation.csv'
###############################################


if __name__ == '__main__':

    # NOTE: Torch represents images as (Channels, Height, Width)
    #       Each sample in the train/test loaders are hence (batch_size, 3, 224, 224)
    # Create the model and train it

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_class = VOC2012(DATASET_DIR, ANNOTATION_CSV)
    model = FCN(n_classes=VOC2012.n_classes)

    train(data_class, model, device)
