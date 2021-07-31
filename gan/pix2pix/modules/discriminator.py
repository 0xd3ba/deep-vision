# discriminator.py -- The discriminator model for Pix2Pix model

import torch
import torch.nn as nn


class ConvNormReLU(nn.Module):
    """
    Module for doing a convolution, followed by a batch norm and finally an activation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, X):
        return self.conv_block(X)


class Discriminator(nn.Module):
    """
    Discriminator for Pix2Pix

    The input to the discriminator are two things:
        - The sketch                                                 (1 channel)
        - The source image (either from generator or the original)   (3 channels)

    Both are concatenated on the channel axis
    """

    def __init__(self, patch_dim=4):
        super().__init__()

        # Input size: (_, 128, 128)
        self.conv_block = nn.Sequential(
            ConvNormReLU(in_channels=1+3, out_channels=64),     # Output: (64, 64, 64)
            ConvNormReLU(in_channels=64, out_channels=128),     # Output: (128, 32, 32)
            ConvNormReLU(in_channels=128, out_channels=256),    # Output: (256, 16, 16)
        )

        # Need to classify each (patch_dim, patch_dim) patch from the input image
        self.patch_disc = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(patch_dim, patch_dim), stride=(patch_dim, patch_dim)),
            nn.Sigmoid()
        )

    def forward(self, X_sketch, X_src):
        """
        X: (batch, channels, height, width)
        """
        batch_size = X_src.shape[0]
        X = torch.cat([X_src, X_sketch], dim=1)

        y_conv = self.conv_block(X)
        y_patch = self.patch_disc(y_conv)

        y_patch = y_patch.reshape(batch_size, -1).mean(dim=-1)
        return y_patch
