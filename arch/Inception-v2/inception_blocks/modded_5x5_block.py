# modded_5x5_block.py -- Class containing the modified Inception block where the 5x5 convolution is
#                        replaced by two 3x3 convolutions

import torch
import torch.nn as nn


class InceptionModded5x5(nn.Module):
    """
    Class implementing the (improvised) Inception module
    that breaks the 5x5 convolution (in the original version) into a series of
    two 3x3 convolutions

    NOTE: The "supplementary material" mentioned in the paper, that describes the
          actual configuration of the modules inside the inception blocks COULD NOT BE FOUND.
          (Pretty horrible way to be honest -- What's so difficult in describing it like GoogLeNet did ? >:( )

          Only mentioned is the aggregated output of the inception module.
          So the way I'm implementing it, i.e. the filter sizes of each parameter, is based on pure speculation
    """
    def __init__(self, in_channels, n_1x1, n_3x3_reduce, n_3x3, n_5x5_reduce, n_5x5, n_pool_proj):
        super().__init__()
        self.conv_3x3_reduce = nn.Conv2d(in_channels=in_channels, out_channels=n_3x3_reduce, kernel_size=(1, 1))
        self.conv_5x5_reduce = nn.Conv2d(in_channels=in_channels, out_channels=n_5x5_reduce, kernel_size=(1, 1))
        self.pool_proj = nn.Conv2d(in_channels=in_channels, out_channels=n_pool_proj, kernel_size=(1, 1))

        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=n_1x1, kernel_size=(1, 1))
        self.conv_3x3 = nn.Conv2d(in_channels=n_3x3_reduce, out_channels=n_3x3, kernel_size=(3, 3), padding='same')

        # The following two layers are equivalent to a 5x5 convolution, when done in series
        self.conv_5x5_p1 = nn.Conv2d(in_channels=n_5x5_reduce, out_channels=n_5x5_reduce, kernel_size=(3, 3), padding='same')
        self.conv_5x5_p2 = nn.Conv2d(in_channels=n_5x5_reduce, out_channels=n_5x5, kernel_size=(3, 3), padding='same')

        # NOTE: Similar to GoogLeNet, Need the output dimensions to match when doing max-pooling
        #       with the input dimensions. This can be achieved by adding a padding of (kernel_size // 2)
        #       Also note that the stride must be (1, 1) for this to happen or else the output dims will be halved
        #       if used the default stride of (2, 2)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, X):
        """
        X: (batch_size, channels, height, width)
        """
        # It is important to note that the outputs of the inception module only differ in the channel dimension
        # Need to concatenate along the channel dimension to produce the required output
        y_1x1 = self.conv_1x1(X)                        # Output shape: (batch, n_1x1, height, width)
        y_3x3 = self.conv_3x3(self.conv_3x3_reduce(X))  # Output shape: (batch, n_3x3, height, width)
        y_maxpool = self.pool_proj(self.max_pool(X))    # Output shape: (batch, pool_proj, height, width)

        y_5x5 = self.conv_5x5_reduce(X)                 # Output shape: (batch, n_5x5_reduce, height, width)
        y_5x5 = self.conv_5x5_p1(y_5x5)                 # Output shape: (batch, n_5x5_reduce, height, width)
        y_5x5 = self.conv_5x5_p2(y_5x5)                 # Output shape: (batch, n_5x5, height, width)

        # Concatenate all the outputs along the channel dimension
        # Doesn't matter in the order they are concatenated
        y_inception = torch.cat([y_1x1, y_3x3, y_5x5, y_maxpool], dim=1)
        return y_inception
