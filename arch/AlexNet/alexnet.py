# alexnet.py -- Module containing the class for the AlexNet model

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Module for the AlexNet model on ImageNette (320px) dataset (ImageNet is too big: ~155GB !! )
    Paper: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

    NOTE: - The dimensions for each image is (3, 227, 227)
            The paper mentions them as (3, 224, 224) which is a mistake

          - The original ImageNet dataset has 1000 classes. ImageNette-320, which is used here,
            has 10 classes. So output layer will have only 10 units, instead of 1000.
    """

    n_classes = 10      # The number of classes in the dataset that is being used

    def __init__(self):
        super().__init__()

        self.conv_layers = [
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),    # Output size: (96, 55, 55)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),          # Output size: (96, 27, 27)

            nn.Conv2d(96, 256, kernel_size=(5, 5), padding='same'),   # Output size: (256, 27, 27)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),          # Output size: (256, 13, 13)

            # Now comes convolutions without any max-pooling
            # The shape remains the same however
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding='same'),  # Output size: (384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding='same'),  # Output size: (384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding='same'),  # Output size: (256, 13, 13)
            nn.ReLU(),

            # The final max-pool layer
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))           # Output size: (256, 6, 6)
        ]

        # Each sample is of shape (256*6*6, ) = (9216, )
        self.fc_layers = [
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),      # AlexNet uses dropout with probability 0.5
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.n_classes)
        ]

        self.conv_net = nn.Sequential(*self.conv_layers)
        self.fc_net = nn.Sequential(*self.fc_layers)

    def forward(self, X):
        """
        X: (batch_size, 3, 227, 227)
        """
        batch_size = X.shape[0]
        y = self.conv_net(X)
        y = y.reshape(batch_size, -1)   # Reshape to match input size of fully connected layers
        y = self.fc_net(y)

        # Not using Softmax activation because Cross-Entropy loss requires un-normalized scores
        return y
