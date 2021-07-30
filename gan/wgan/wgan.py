# wgan.py -- Module containing the classes for Generator and Discriminator models based
#            on CNNs

# One small change in Generator: Instead of using Tanh in the output layer, I'm using
# Sigmoid. This is because, in my case, the input images are normalized to [0, 1] range

import torch
import torch.nn as nn

class GeneratorCNN(nn.Module):
    """
    Module for Generator Model
    """

    class UpSampleBlock(nn.Module):
        """ Up-sampling block for the generator """
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2), stride=(2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=in_channels),

                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=out_channels)
            )

        def forward(self, X):
            """ X: (channels, height, width) """
            return self.conv_block(X)

    def __init__(self, in_dims, out_dims):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims

        self.fc = nn.Linear(in_dims, 16*4*4)    # This will be reshaped to (16, 4, 4)

        # Input shape:   (16, 4, 4)
        # Output shape:  (3, 64, 64), i.e the same shape as true images
        self.conv_t_block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1)),  # Output: (64, 4, 4)
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            # Start the up-sampling operation
            self.UpSampleBlock(in_channels=64, out_channels=64),   # Output: (64, 8, 8)
            self.UpSampleBlock(in_channels=64, out_channels=32),   # Output: (32, 16, 16)
            self.UpSampleBlock(in_channels=32, out_channels=16),   # Output: (16, 32, 32)
            self.UpSampleBlock(in_channels=16, out_channels=8),    # Output: (8, 64, 64)

            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(1, 1)),  # Output: (3, 64, 64)
            nn.Sigmoid()
        )

    def forward(self, X):
        """
        X:      (batch, in_dims)
        Output: (batch, 3, 64, 64)
        """
        y_fc = self.fc(X)
        y_fc = y_fc.reshape(-1, 16, 4, 4)     # Reshape before feeding to convolutional block
        y_op = self.conv_t_block(y_fc)

        return y_op


class CriticCNN(nn.Module):
    """
    Module for Critic Model (Basiscally the same thing as Discriminator, only difference is the absence of Sigmoid)

    The last layer simply predicts a score, instead of probability
    """

    class DownSampleBlock(nn.Module):
        """ Module for downsampling block """

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding='same'),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=in_channels),

                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=(2, 2)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=out_channels)
            )

        def forward(self, X):
            """ X: (batch, channels, height, width) """
            return self.conv_block(X)

    def __init__(self, out_dims=1):
        super().__init__()

        self.out_dims = out_dims

        # Input shape: (3, 64, 64)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(1, 1)),   # Output: (8, 64, 64)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=8),

            self.DownSampleBlock(in_channels=8, out_channels=16),           # Output: (16, 32, 32)
            self.DownSampleBlock(in_channels=16, out_channels=32),          # Output: (32, 16, 16)
            self.DownSampleBlock(in_channels=32, out_channels=64),          # Output: (64, 8, 8)
            self.DownSampleBlock(in_channels=64, out_channels=64)           # Output: (64, 4, 4)
        )

        # Input: (64*4*4, )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 1),

            # No sigmoid activation used here, unlike DCGAN
        )

    def forward(self, X):
        """
        X:      (batch, in_dims)
        Output: (batch, out_dims)
        """
        batch_size = X.shape[0]
        y_conv = self.conv_block(X)
        y_conv = y_conv.reshape(batch_size, -1)
        y_fc = self.fc(y_conv)

        return y_fc
