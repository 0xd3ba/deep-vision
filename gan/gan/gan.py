# gan.py -- Module containing the classes for Generator and Discriminator models

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Module for Generator Model

    Implemented as a simple feed-forward neural net
    """

    def __init__(self, in_dims, out_dims):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims

        self.model = nn.Sequential(
            nn.Linear(in_dims, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, out_dims)
        )

    def forward(self, X):
        """
        X:      (batch, in_dims)
        Output: (batch, out_dims)
        """
        return self.model(X)


class Discriminator(nn.Module):
    """
    Module for Discriminator Model

    Implemented as a simple feed-forward neural net.
    The last layer simply predicts a probability that the given input is real (label=1)
    or fake (label=0)
    """

    def __init__(self, in_dims, out_dims=1):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims

        self.model = nn.Sequential(
            nn.Linear(in_dims, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, out_dims),

            nn.Sigmoid()        # Because Binary Cross-Entropy loss will be used
        )

    def forward(self, X):
        """
        X:      (batch, in_dims)
        Output: (batch, out_dims)
        """
        return self.model(X)
