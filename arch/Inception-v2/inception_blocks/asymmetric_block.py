# asymmetric_block.py -- Class containing the Inception block that replaces a (n x n) convolution
#                        with multiple asymmetric (1 x n) or (n x 1) blocks

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionAsymmetric7x7(nn.Module):
    """
    Class containing the inception block that contains asymmetric 7x7 convolutions.

    NOTE: Two 7x7 convolutions IN SERIES is equivalent to a single 13x13 convolution
          In general, two nxn convolutions in series is equivalent to a single (2n-1)x(2n-1) convolution

    """
    def __init__(self, in_channels, n_1x1, n_7x7_reduce, n_7x7, n_13x13_reduce, n_13x13, n_pool_proj):
        super().__init__()

        self.conv_7x7_reduce = nn.Conv2d(in_channels=in_channels, out_channels=n_7x7_reduce, kernel_size=(1, 1))
        self.conv_13x13_reduce = nn.Conv2d(in_channels=in_channels, out_channels=n_13x13_reduce, kernel_size=(1, 1))
        self.pool_proj = nn.Conv2d(in_channels=in_channels, out_channels=n_pool_proj, kernel_size=(1, 1))

        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=n_1x1, kernel_size=(1, 1))
        self.conv_7x7_p1 = nn.Conv2d(in_channels=n_7x7_reduce, out_channels=n_7x7_reduce, kernel_size=(1, 7), padding='same')
        self.conv_7x7_p2 = nn.Conv2d(in_channels=n_7x7_reduce, out_channels=n_7x7, kernel_size=(7, 1), padding='same')

        self.conv_13x13_p1 = nn.Conv2d(in_channels=n_13x13_reduce, out_channels=n_13x13_reduce, kernel_size=(1, 7), padding='same')
        self.conv_13x13_p2 = nn.Conv2d(in_channels=n_13x13_reduce, out_channels=n_13x13_reduce, kernel_size=(7, 1), padding='same')
        self.conv_13x13_p3 = nn.Conv2d(in_channels=n_13x13_reduce, out_channels=n_13x13_reduce, kernel_size=(1, 7), padding='same')
        self.conv_13x13_p4 = nn.Conv2d(in_channels=n_13x13_reduce, out_channels=n_13x13, kernel_size=(7, 1), padding='same')

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y_1x1 = self.conv_1x1(X)                     # Output shape: (batch, n_1x1, height, width)

        y_7x7 = self.conv_7x7_reduce(X)
        y_7x7 = F.relu(y_7x7)
        y_7x7 = self.conv_7x7_p1(y_7x7)
        y_7x7 = self.conv_7x7_p2(y_7x7)              # Output shape: (batch, n_7x7, height, width)

        y_13x13 = self.conv_13x13_reduce(X)
        y_13x13 = F.relu(y_7x7)
        y_13x13 = self.conv_13x13_p1(y_13x13)
        y_13x13 = self.conv_13x13_p2(y_13x13)
        y_13x13 = self.conv_13x13_p3(y_13x13)
        y_13x13 = self.conv_13x13_p4(y_13x13)        # Output shape: (batch, n_13x13, height, width)

        y_pool = self.pool_proj( self.max_pool(X) )  # Output shape: (batch, n_pool_proj, height, width)

        # Finally concatentate everything on the channel dimension
        y_concat = torch.cat([y_1x1, y_7x7, y_13x13, y_pool], dim=1)

        return y_concat
