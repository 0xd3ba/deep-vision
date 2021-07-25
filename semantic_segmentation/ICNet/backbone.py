# backbone.py -- Module containing the backbone for the ICNet model
#
# NOTE: I'm using a simple UNet-style encoder network for this


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolution Block containing a series of 'n' convolutions along with Batch Normalization
    Number of kernel maps to use depends on the supplied parameter
    """

    def __init__(self, in_channels, out_channels, n_layers=2, kernel_size=(3, 3), downsample=True, dilation=(1, 1)):
        super().__init__()
        self.n_kernels = n_layers
        self.ip_channels = in_channels
        self.op_channels = out_channels

        self.conv_layers = []
        curr_channel = in_channels

        # Add the convolution layers with same padding, i.e. output is same as input shape
        # before max-pooling operation is done
        padding = 'same'

        for i in range(n_layers):

            # We dont want output of convolution operation to be the same as the input
            # size when we want a strided-convolution. Need a padding of (1, 1) to ensure dimensions
            # match with the 1x1 convolution of the residual connection (with stride (2,2))

            conv_layer_i = nn.Conv2d(in_channels=curr_channel,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=padding)
            curr_channel = out_channels
            self.conv_layers.append(conv_layer_i)
            self.conv_layers.append(nn.BatchNorm2d(out_channels))   # Batch Normalization
            self.conv_layers.append(nn.ReLU())

        # Finally add the Max-Pool layer if we need it
        if downsample:
            self.conv_layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # Build the block as a model
        self.conv_block = nn.Sequential(*self.conv_layers)

    def forward(self, X):
        """
        X:  (batch, n_channels, height, width)
        """
        return self.conv_block(X)


class EncoderNetwork(nn.Module):
    """
    Module for the UNet-style encoder network
    Uses 3x3 convolutions and pool layers of (2, 2) kernels with stride of (2,2)
    """

    def __init__(self, channels_list, use_dilations=False, dilation_rates=None, dilation_op_channels=None):
        super().__init__()

        # NOTE: The image is down-sampled only 3 times, as mentioned in the paper
        #       But still made this quite generic to support down-sampling any valid number of times

        self.conv_layers = []
        curr_channels = 3
        for i in range(len(channels_list)):
            conv_blk = ConvBlock(in_channels=curr_channels, out_channels=channels_list[i], n_layers=2)
            self.conv_layers.append(conv_blk)
            curr_channels = channels_list[i]

        # If we need to use Dilated convolutions
        # Needed in only the first layer of the ICNet architecture
        if use_dilations:

            assert dilation_op_channels is not None, "Dilation output channels must not be None"
            assert len(dilation_rates) == len(dilation_op_channels), \
                "# of Dilation output channels must match the # of dilation rates"

            curr_channels = channels_list[-1]
            for i in range(len(dilation_op_channels)):
                self.conv_layers.append(
                    ConvBlock(in_channels=curr_channels,
                              out_channels=dilation_op_channels[i],
                              downsample=False,
                              dilation=(dilation_rates[i], dilation_rates[i]))
                )

                curr_channels = dilation_op_channels[i]

            # Now need to add the 1x1 convolution to reduce the channel dimensions
            # Set it to half the number of last dilation output channel
            self.conv_layers.append(
                nn.Conv2d(in_channels=dilation_op_channels[-1],
                          out_channels=dilation_op_channels[-1]//2,
                          kernel_size=(1, 1))
            )

        # Build the model
        self.conv_block = nn.Sequential(*self.conv_layers)

    def forward(self, X):
        """
        X:  (batch, channels, height, width)
        """
        return self.conv_block(X)
