# depthwise_separable.py -- Module containing the class for the depthwise-separable convolutions

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Class for Depthwise-Separable convolution

    NOTE: The padding must be set to "same", i.e. dimensions remain the same after the convolution
          or else the shape obtained after last layer doesn't match what is mentioned in the paper !
          This does not happen when padding is set accordingly. So padding is necessary.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        # NOTE: The groups must be equivalent to the number of input channels so that
        #       each channel gets its own kernel filter. The output channels must also be equal
        #       to the input channels because each kernel filter produces a single output channel
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=kernel_size, groups=in_channels, padding='same')
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y = self.depthwise_conv(X)
        y = self.pointwise_conv(X)
        y = self.batch_norm(y)

        return y