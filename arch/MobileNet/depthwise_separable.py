# depthwise_separable.py -- Module containing the class for the depthwise-separable convolutions

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Class for Depthwise-Separable convolution

    NOTE: The padding must be set to "same", i.e. dimensions remain the same after the convolution
          or else the shape obtained after last layer doesn't match what is mentioned in the paper !
          This does not happen when padding is set accordingly. So padding is necessary.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1)):
        super().__init__()

        # NOTE: The groups must be equivalent to the number of input channels so that
        #       each channel gets its own kernel filter. The output channels must also be equal
        #       to the input channels because each kernel filter produces a single output channel

        if stride != (1, 1):
            padding = (1, 1)        # Need this to match the shapes that are mentioned in the paper
        else:
            padding = 'same'

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=stride,
                                        kernel_size=kernel_size, groups=in_channels, padding=padding)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.batch_norm_depthwise = nn.BatchNorm2d(num_features=in_channels)
        self.batch_norm_pointwise = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y = self.depthwise_conv(X)
        y = self.batch_norm_depthwise(y)
        y = F.relu(y)

        y = self.pointwise_conv(y)
        y = self.batch_norm_pointwise(y)
        y = F.relu(y)

        return y
