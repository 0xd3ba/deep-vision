# entry_flow.py -- Module containing the class for the entry-flow of the Xception architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.depthwise_separable import DepthwiseSeparableConv


class EntryFlowSeparableConv(nn.Module):
    """
    Class for depthwise-separable convolutions for entry-flow part of the model
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

    def forward(self, X):
        y = self.block(X)
        return y


class XceptionEntryFlow(nn.Module):
    """
    Entry flow of Xception Architecture.

    There are a few minor differences with what the paper presents and what I'm getting
    however. Might be because of internal implementation differences between Tensorflow/Keras
    and PyTorch.

    Shape of each input:  (3, 299, 299)
    Shape of each output: (728, 19, 19)       (what paper presents)
                          (728, 17, 17)       (what I'm getting o.O)

    Also note that because of these differences, the residual connection uses (3, 3) convolutions
    instead of (1, 1) or else the dimensions doesn't match up. Small differences but I don't think
    these would impact the final result significantly,
    """

    def __init__(self):
        super().__init__()

        # Block-1 containing simple convolutions
        self.conv_blk_1 = nn.Sequential(                                                    # Shape: (3, 299, 299)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),   # Shape: (32, 149, 149)
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),                 # Shape: (64, 147, 147)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
         )

        # NOTE: There is one extra RELU being done for the first block, which doesn't matter anyways
        #       because of the nature of the activation function
        # The block contains two (3, 3) convolutions with stride of (1, 1) which is equivalent to a
        # (5, 5) convolution, which is followed by a max-pool operation with kernel size (3, 3) and stride of (2, 2)
        # Because of padding, the output of the convolution remains the same.

        # Input shape:  (64, 147, 147)
        # Output shape: (128, 147, 147) -- maxpool --> (128, 73, 73)
        self.residual_blk_1_to_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        self.conv_blk_2 = EntryFlowSeparableConv(in_channels=64, out_channels=128, kernel_size=(3, 3))

        # Input shape:  (128, 73, 73)
        # Output shape: (256, 73, 73) -- maxpool --> (128, 36, 36)
        self.residual_blk_2_to_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
        self.conv_blk_3 = EntryFlowSeparableConv(in_channels=128, out_channels=256, kernel_size=(3, 3))

        # Input shape:  (256, 36, 36)
        # Output shape: (728, 36, 36) -- maxpool --> (728, 17, 17)
        self.residual_blk_3_to_4 = nn.Conv2d(in_channels=256, out_channels=728, kernel_size=(3, 3), stride=(2, 2))
        self.conv_blk_4 = EntryFlowSeparableConv(in_channels=256, out_channels=728, kernel_size=(3, 3))

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y = self.conv_blk_1(X)
        y = self.conv_blk_2(y) + self.residual_blk_1_to_2(y)
        y = self.conv_blk_3(y) + self.residual_blk_2_to_3(y)
        y = self.conv_blk_4(y) + self.residual_blk_3_to_4(y)    # Output shape: (batch, 728, 17, 17)

        return y
