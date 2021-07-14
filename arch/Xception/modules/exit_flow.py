# exit_flow.py -- Module containing the middle-flow of the Xception architecture

import torch
import torch.nn as nn
from modules.depthwise_separable import DepthwiseSeparableConv


class XceptionExitFlow(nn.Module):
    """
    Middle flow of Xception Architecture.

    There are a few minor differences with what the paper presents and what I'm getting
    however. Might be because of internal implementation differences between Tensorflow/Keras
    and PyTorch.

    Shape of each input:  (728, 19, 19)       (what paper presents)
                          (728, 17, 17)       (what I'm getting o.O)

    Also note that because of these differences, the residual connection uses (3, 3) convolutions
    instead of (1, 1) or else the dimensions doesn't match up. Small differences but I don't think
    these would impact the final result significantly,
    """

    n_repeat = 8    # There are 8 depthwise-separable blocks in total, in the middle flow

    def __init__(self):
        super().__init__()

        # Input shape:  (728, 17, 17)
        # Output shape: (1024, 17, 17) -- maxpool --> (1024, 8, 8)
        self.residual = nn.Conv2d(in_channels=728, out_channels=1024, kernel_size=(3, 3), stride=(2, 2))
        self.conv_blk_1 = nn.Sequential(
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=728, out_channels=728, kernel_size=(3, 3)),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=728, out_channels=1024, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

        # Input shape:  (1024, 8, 8)
        # Output shape: (2048, 1, 1)
        self.conv_blk_2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=1024, out_channels=1536, kernel_size=(3, 3)),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=1536, out_channels=2048, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(8, 8), stride=(1, 1))
        )

    def forward(self, X):
        """
         X: (batch, channels, height, width)
        """
        y = self.conv_blk_1(X) + self.residual(X)
        y = self.conv_blk_2(y)                      # Output shape: (batch, 2048, 1, 1)

        return y
