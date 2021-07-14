# xception.py -- Module containing the class for the Xception model


import torch
import torch.nn as nn
from modules.entry_flow import XceptionEntryFlow
from modules.middle_flow import XceptionMiddleFlow
from modules.exit_flow import XceptionExitFlow


class Xception(nn.Module):
    """
    Module for the Xception model on ImageNette (320px) dataset (ImageNet is too big: ~155GB !! )
    Paper: https://arxiv.org/abs/1610.02357

    NOTE: - The original ImageNet dataset has 1000 classes. ImageNette-320, which is used here,
            has 10 classes. So output layer will have only 10 units, instead of 1000.
    """

    n_classes = 10      # The number of classes in the dataset that is being used

    def __init__(self):
        super().__init__()

        self.conv_layers = [       # Input shape:  (3, 299, 299)
            XceptionEntryFlow(),   # Output shape: (728, 17, 17)
            XceptionMiddleFlow(),  # Output shape: (728, 17, 17)
            XceptionExitFlow()     # Output shape: (2048, 1, 1)
        ]

        # Each sample is of shape (2048*1*1, ) = (2048, )
        self.fc_layers = [
            nn.Linear(2048, self.n_classes)
        ]

        self.conv_net = nn.Sequential(*self.conv_layers)
        self.fc_net = nn.Sequential(*self.fc_layers)

    def forward(self, X):
        """
        X: (batch_size, 3, 227, 227)
        """
        batch_size = X.shape[0]
        y = self.conv_net(X)
        y = y.reshape(batch_size, -1)  # Reshape to match input size of fully connected layers
        y = self.fc_net(y)

        # Not using Softmax activation because Cross-Entropy loss requires un-normalized scores
        return y
