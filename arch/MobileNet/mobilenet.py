# mobilenet.py -- Module containing the class for the MobileNet model


import torch
import torch.nn as nn
from depthwise_separable import DepthwiseSeparableConv


class ConvBNormActivation(nn.Module):
    """
    Module that takes care of convolution followed by batch norm and then a ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, X):
        return self.block(X)


class MobileNet(nn.Module):
    """
    Module for the MobileNet model on ImageNette (320px) dataset (ImageNet is too big: ~155GB !! )
    Paper: https://arxiv.org/abs/1704.04861

    NOTE: - The original ImageNet dataset has 1000 classes. ImageNette-320, which is used here,
            has 10 classes. So output layer will have only 10 units, instead of 1000.
    """

    n_classes = 10      # The number of classes in the dataset that is being used

    def __init__(self):
        super().__init__()

        # Note: Need to add some padding to match the shapes as described in the paper

        self.conv_layers = [                                                                                    # Input shape:  (3, 224, 224)
            ConvBNormActivation(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),  # Output shape: (32, 112, 112)
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=(3, 3)),                        # Output shape: (32, 112, 112)

            ConvBNormActivation(in_channels=32, out_channels=64, kernel_size=(1, 1)),                           # Output shape: (64, 112, 112)
            DepthwiseSeparableConv(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),         # Output shape: (64, 56, 56)

            ConvBNormActivation(in_channels=64, out_channels=128, kernel_size=(1, 1)),                          # Output shape: (128, 56, 56)
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=(3, 3)),                      # Output shape: (128, 56, 56)

            ConvBNormActivation(in_channels=128, out_channels=128, kernel_size=(1, 1)),                         # Output shape: (128, 56, 56)
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2)),       # Output shape: (128, 28, 28)

            ConvBNormActivation(in_channels=128, out_channels=256, kernel_size=(1, 1)),                         # Output shape: (256, 28, 28)
            DepthwiseSeparableConv(in_channels=256, out_channels=256, kernel_size=(3, 3)),                      # Output shape: (128, 28, 28)

            ConvBNormActivation(in_channels=256, out_channels=256, kernel_size=(1, 1)),                         # Output shape: (256, 28, 28)
            DepthwiseSeparableConv(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2)),       # Output shape: (256, 14, 14)

            ConvBNormActivation(in_channels=256, out_channels=512, kernel_size=(1, 1)),                         # Output shape: (512, 14, 14)

            # depth-wise followed by (1,1) convolution repeats 5 times
            DepthwiseSeparableConv(in_channels=512, out_channels=512, kernel_size=(3, 3)),                      # Output shape: (512, 14, 14)
            ConvBNormActivation(in_channels=512, out_channels=512, kernel_size=(1, 1)),                         # Output shape: (512, 14, 14)
            DepthwiseSeparableConv(in_channels=512, out_channels=512, kernel_size=(3, 3)),                      # Output shape: (512, 14, 14)
            ConvBNormActivation(in_channels=512, out_channels=512, kernel_size=(1, 1)),                         # Output shape: (512, 14, 14)
            DepthwiseSeparableConv(in_channels=512, out_channels=512, kernel_size=(3, 3)),                      # Output shape: (512, 14, 14)
            ConvBNormActivation(in_channels=512, out_channels=512, kernel_size=(1, 1)),                         # Output shape: (512, 14, 14)
            DepthwiseSeparableConv(in_channels=512, out_channels=512, kernel_size=(3, 3)),                      # Output shape: (512, 14, 14)
            ConvBNormActivation(in_channels=512, out_channels=512, kernel_size=(1, 1)),                         # Output shape: (512, 14, 14)
            DepthwiseSeparableConv(in_channels=512, out_channels=512, kernel_size=(3, 3)),                      # Output shape: (512, 14, 14)
            ConvBNormActivation(in_channels=512, out_channels=512, kernel_size=(1, 1)),                         # Output shape: (512, 14, 14)

            DepthwiseSeparableConv(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2)),       # Output shape: (512, 7, 7)
            ConvBNormActivation(in_channels=512, out_channels=1024, kernel_size=(1, 1)),                        # Output shape: (1024, 7, 7)

            # One small difference: Using a stride of (2,2) below is giving me output of shape (1024, 4, 4)
            # Not using (2,2) stride here hence.
            DepthwiseSeparableConv(in_channels=1024, out_channels=1024, kernel_size=(3, 3)),                    # Output shape: (1024, 7, 7)
            ConvBNormActivation(in_channels=1024, out_channels=1024, kernel_size=(1, 1)),                       # Output shape: (1024, 7, 7)

            # Finally, the average pooling layer
            nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))   # Output shape: (1024, 1, 1)
        ]

        # Each sample is of shape (1024*1*1, ) = (1024, )
        self.fc_layers = [
            nn.Linear(1024, 1000),
            nn.Linear(1000, self.n_classes)
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
