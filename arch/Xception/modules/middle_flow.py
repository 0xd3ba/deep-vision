# middle_flow.py -- Module containing the middle-flow of the Xception architecture

import torch
import torch.nn as nn
from modules.depthwise_separable import DepthwiseSeparableConv


class MiddleFlowSeparableConv(nn.Module):
    """
    Class for depthwise-separable convolutions for middle-flow part of the model
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
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=kernel_size)
        )

    def forward(self, X):
        y = self.block(X)
        return y


class XceptionMiddleFlow(nn.Module):
    """
    Middle flow of Xception Architecture.

    There are a few minor differences with what the paper presents and what I'm getting
    however. Might be because of internal implementation differences between Tensorflow/Keras
    and PyTorch.

    Shape of each input:  (728, 19, 19)       (what paper presents)
                          (728, 17, 17)       (what I'm getting)

    Shape of each output: (728, 19, 19)       (what paper presents)
                          (728, 17, 17)       (what I'm getting o.O)

    Also note that because of these differences, the residual connection uses (3, 3) convolutions
    instead of (1, 1) or else the dimensions doesn't match up. Small differences but I don't think
    these would impact the final result significantly,
    """

    n_repeat = 8    # There are 8 depthwise-separable blocks in total, in the middle flow

    def __init__(self):
        super().__init__()
        self.conv_blks = [
            MiddleFlowSeparableConv(in_channels=728, out_channels=728, kernel_size=(3, 3))
            for _ in range(self.n_repeat)
        ]

    def forward(self, X):
        """
        X: (batch, channels, height, width)

        NOTE: Because feature dimensions match, there is no need for transformation
              on residual connections. This makes things simpler as the entire thing
              can be written pretty neatly
        """
        y = X
        for conv_blk in self.conv_blks:
            y = conv_blk(y) + y

        return y