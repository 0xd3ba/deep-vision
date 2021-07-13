# coarse_block.py -- Class containing the final incception block in the Inception-v2 network
#                    I don't know what to call this block specifically, but it doesn't matter anyways

import torch
import torch.nn as nn


class InceptionCoarse(nn.Module):
    """
    Architecture of the final inception block in the network

    NOTE:
        - n_3x1_3 refers to the # of output channels in the left-most block of Figure(7) in the paper
        - n_1x1_3 refers to the # of output channels in the second block from the left, i.e. to the right of above block
    """
    def __init__(self, in_channels, n_1x1, n_3x1_3_reduce, n_3x1_3, n_1x1_3_reduce, n_1x1_3, n_pool_proj):
        super().__init__()

        self.conv_3x1_3_reduce = nn.Conv2d(in_channels=in_channels, out_channels=n_3x1_3_reduce, kernel_size=(1, 1))
        self.conv_1x1_3_reduce = nn.Conv2d(in_channels=in_channels, out_channels=n_1x1_3_reduce, kernel_size=(1, 1))
        self.pool_proj = nn.Conv2d(in_channels=in_channels, out_channels=n_pool_proj, kernel_size=(1, 1))

        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=n_1x1, kernel_size=(1, 1))

        self.conv_3x1_3 = nn.Conv2d(in_channels=n_3x1_3_reduce, out_channels=n_3x1_3_reduce, kernel_size=(3, 3), padding='same')
        self.conv_3x1_3_p1 = nn.Conv2d(in_channels=n_3x1_3_reduce, out_channels=(n_3x1_3 // 2), kernel_size=(1, 3), padding='same')
        self.conv_3x1_3_p2 = nn.Conv2d(in_channels=n_3x1_3_reduce, out_channels=(n_3x1_3 - (n_3x1_3 // 2)), kernel_size=(3, 1), padding='same')

        self.conv_1x1_3_p1 = nn.Conv2d(in_channels=n_1x1_3_reduce, out_channels=(n_1x1_3 // 2), kernel_size=(1, 3), padding='same')
        self.conv_1x1_3_p2 = nn.Conv2d(in_channels=n_1x1_3_reduce, out_channels=(n_1x1_3 - (n_1x1_3 // 2)), kernel_size=(3, 1), padding='same')

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y_1x1 = self.conv_1x1(X)

        y_3x1_3 = self.conv_3x1_3_reduce(X)
        y_3x1_3 = self.conv_3x1_3(y_3x1_3)
        y_3x1_3_p1 = self.conv_3x1_3_p1(y_3x1_3)    # This is used in concatenation
        y_3x1_3_p2 = self.conv_3x1_3_p2(y_3x1_3)    # This is used in concatenation

        y_1x1_3 = self.conv_1x1_3_reduce(X)
        y_1x1_3_p1 = self.conv_1x1_3_p1(y_1x1_3)    # This is used in concatenation
        y_1x1_3_p2 = self.conv_1x1_3_p2(y_1x1_3)    # This is used in concatenation

        y_pool = self.pool_proj( self.max_pool(X) )

        # Finally concatenate the relevant outputs and return
        # Doesn't matter in the order they are concatenated
        y_cat = torch.cat([y_3x1_3_p1, y_3x1_3_p2, y_1x1_3_p1, y_1x1_3_p2, y_pool, y_1x1], dim=1)

        return y_cat

