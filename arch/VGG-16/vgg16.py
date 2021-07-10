# vgg26.py -- Module containing the class for the VGG-16 model


import torch
import torch.nn as nn


class ConvBlock3x3(nn.Module):
    """
    Convolution Block containing 3x3 kernel maps and a final 2x2 max-pooling layer.
    Number of kernel maps to use depends on the supplied parameter

    NOTE: The paper mentions about using padded convolutions such that the result
          after convolution is the same as the input before the convolution occurred
    """

    def __init__(self, n_layers, in_channels, out_channels):
        super().__init__()
        self.n_kernels = n_layers
        self.ip_channels = in_channels
        self.op_channels = out_channels

        self.conv_layers = []
        curr_channel = in_channels

        # Add the convolution layers
        for i in range(n_layers):
            conv_layer_i = nn.Conv2d(in_channels=curr_channel,
                                     out_channels=out_channels,
                                     kernel_size=(3, 3),
                                     padding='same')
            curr_channel = out_channels
            self.conv_layers.append(conv_layer_i)

        # Finally add the max-pool layer
        pool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv_layers.append(pool_layer)

        # Build the block as a model
        self.conv_block = nn.Sequential(*self.conv_layers)

    def forward(self, X):
        """
        X:  (batch, n_channels, height, width)
        """
        return self.conv_block(X)


class VGG16(nn.Module):
    """
    Module for the VGG-16 model on ImageNette (320px) dataset (ImageNet is too big: ~155GB !! )
    Paper: https://arxiv.org/abs/1409.1556

    NOTE: - The original ImageNet dataset has 1000 classes. ImageNette-320, which is used here,
            has 10 classes. So output layer will have only 10 units, instead of 1000.
    """

    n_classes = 10      # The number of classes in the dataset that is being used

    def __init__(self):
        super().__init__()

        # Build the convolution blocks as mentioned in the paper
        self.conv_blocks = [                                                # Input shape:  (3, 224, 224)
            ConvBlock3x3(n_layers=2, in_channels=3, out_channels=64),       # Output shape: (64, 112, 112)
            ConvBlock3x3(n_layers=2, in_channels=64, out_channels=128),     # Output shape: (128, 56, 56)
            ConvBlock3x3(n_layers=2, in_channels=128, out_channels=256),    # Output shape: (256, 28, 28)

            # Now comes the blocks with 3 convolution layers
            ConvBlock3x3(n_layers=3, in_channels=256, out_channels=512),    # Output shape: (512, 14, 14)
            ConvBlock3x3(n_layers=3, in_channels=512, out_channels=512)     # Output shape: (512, 7, 7)
        ]

        # Input to the fully-connected network is (512*7*7, ) = (25088, )
        self.fc_layers = [
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.n_classes)
        ]

        # Finally build the model
        self.conv_blocks_model = nn.Sequential(*self.conv_blocks)
        self.fc_model = nn.Sequential(*self.fc_layers)

    def forward(self, X):
        """
        X:  (batch, 3, 224, 224)
        """
        batch_size = X.shape[0]
        y = self.conv_blocks_model(X)
        y = y.reshape(batch_size, -1)       # Flatten the output of the convolutional layers
        y = self.fc_model(y)                # Finally get the raw scores for each class. Shape: (batch, n_classes)

        return y
