# pyramid_pool.py -- The pyramid pooling module, as described in the paper

import torch
import torch.nn as nn


class PyramidMaxPool(nn.Module):
    """
    Pyramid Max-Pooling Module that applies max-pooling operation to produce
    feature maps of sizes:
        - (channels, 1,1)
        - (channels, 2,2)
        - (channels, 3,3)
        - (channels, 6,6)

    Then applies a (1,1) convolution to decrease the input channels to (channels // n_pyramid)
    Then directly upsamples them with bilinear-interpolation to match the input shape
    Finally they are concatenated with the input in the channel dimension and the result is returned

    NOTE: Here n_pyramid = 4, because we are producing a pyramid with output of 4 pooling layers
    """

    def __init__(self, in_channels, pyramid_kernel_dims):
        super().__init__()

        out_channels = in_channels // len(pyramid_kernel_dims)

        self.pyramid_pools = [nn.AdaptiveMaxPool2d((i, i)) for i in pyramid_kernel_dims]
        self.conv_1x1s = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
            for _ in range(len(pyramid_kernel_dims))
        ]

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y_pools = []
        h, w = X.shape[-2:]     # Height and width of the input

        upsampler = nn.Upsample(size=(h, w), mode='bilinear', align_corners=False)

        # Gather the result of all pyramid pools into y_pools
        for pyramid_pool, conv_1x1 in zip(self.pyramid_pools, self.conv_1x1s):
            y = pyramid_pool(X)
            y = conv_1x1(y)
            y = upsampler(y)        # Up-sample to match the input dimensions

            y_pools.append(y)

        # Concatenate the input with the pool outputs in the channel dimension
        y_final = torch.cat([X, *y_pools], dim=1)
        return y_final
