# pspnet.py -- Module containing the class for the PSPNet model
#              Unlike the original paper, which uses ResNet as the encoder network,
#              I'm choosing to use VGG-16 as the backbone (just like I did in SegNet)
#              because of simplicity.
#
# PSPNet is not a complete network architecture, but an additional layer
# just after the encoder network.

import torch
import torch.nn as nn
from pyramid_pool import PyramidMaxPool


class EncoderBlock3x3(nn.Module):
    """
    Convolution Block containing 3x3 kernel maps and a final 2x2 max-pooling layer.
    Number of kernel maps to use depends on the supplied parameter

    NOTE: This is the VGG-16 convolution block
    """

    def __init__(self, n_layers, in_channels, out_channels):
        super().__init__()
        self.n_kernels = n_layers
        self.ip_channels = in_channels
        self.op_channels = out_channels

        self.conv_layers = []
        curr_channel = in_channels

        # Add the convolution layers
        for i in range(n_layers):
            conv_layer_i = nn.Conv2d(in_channels=curr_channel,
                                     out_channels=out_channels,
                                     kernel_size=(3, 3),
                                     padding='same')
            curr_channel = out_channels
            self.conv_layers.append(conv_layer_i)
            self.conv_layers.append(nn.BatchNorm2d(num_features=out_channels))
            self.conv_layers.append(nn.ReLU())

        # Finally add the max-pool layer
        pool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv_layers.append(pool_layer)

        # Build the block as a model
        self.conv_block = nn.Sequential(*self.conv_layers)

    def forward(self, X):
        """
        X:  (batch, n_channels, height, width)
        """
        return self.conv_block(X)


class DecoderBlock3x3(nn.Module):
    """
    Transposed convolution block with 3x3 kernels and a final 2x2 kernel with stride of 2 (to undo max-pool operation)
    """
    def __init__(self, n_layers, in_channels, out_channels):
        super().__init__()
        self.n_kernels = n_layers
        self.ip_channels = in_channels
        self.op_channels = out_channels

        self.conv_t_layers = []
        curr_channel = in_channels

        # Add the convolution layers
        for i in range(n_layers):
            conv_t_layer_i = nn.ConvTranspose2d(in_channels=curr_channel,
                                                out_channels=out_channels,
                                                kernel_size=(3, 3),
                                                padding=(1, 1))
            curr_channel = out_channels
            self.conv_t_layers.append(conv_t_layer_i)
            self.conv_t_layers.append(nn.BatchNorm2d(num_features=out_channels))
            self.conv_t_layers.append(nn.ReLU())

        # Finally add the unpooling layer that undoes the max-pool operation in encoding layer
        unpool_layer = nn.ConvTranspose2d(in_channels=out_channels,
                                          out_channels=out_channels,
                                          kernel_size=(2, 2),
                                          stride=(2, 2))
        self.conv_t_layers.append(unpool_layer)

        # Build the block as a model
        self.conv_block = nn.Sequential(*self.conv_t_layers)

    def forward(self, X):
        """
        X:  (batch, n_channels, height, width)
        """
        return self.conv_block(X)


class PSPNet(nn.Module):
    """
    Class for the PSPNet model

    Because the backbone is VGG-16, the input images muse be atleast
    (224, 224) with 3 channels (RGB). The images can be greater in any dimension, but can't go less than 224
    """

    # The kernel dimensions for each side, i.e. square kernels
    # These will be the kernels that will be used in the pyramid pooling step
    pyramid_kernel_dims = [1, 2, 3, 6]

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        # I'm assuming (3, 224, 224) image as calculation of output shapes in transposed-convolution
        # is non-trivial. Leads to some rounding and stuff, which means it might produce different shape
        # from the input that was used in the convolution operation previously.

        # PART-1: The Encoder network
        self.encoder_conv_blk_1 = EncoderBlock3x3(n_layers=2, in_channels=3, out_channels=64)      # Output shape: (64, 112, 112)
        self.encoder_conv_blk_2 = EncoderBlock3x3(n_layers=2, in_channels=64, out_channels=128)    # Output shape: (128, 56, 56)
        self.encoder_conv_blk_3 = EncoderBlock3x3(n_layers=2, in_channels=128, out_channels=256)   # Output shape: (256, 28, 28)
        self.encoder_conv_blk_4 = EncoderBlock3x3(n_layers=3, in_channels=256, out_channels=512)   # Output shape: (512, 14, 14)
        self.encoder_conv_blk_5 = EncoderBlock3x3(n_layers=3, in_channels=512, out_channels=512)   # Output shape: (512, 7, 7)

        # The Pyramid-pooling layer -- heart of the paper
        enc_op_channels = 512
        n_pyramid_kernels = len(self.pyramid_kernel_dims)
        pyramid_out_channels = enc_op_channels + ((enc_op_channels // n_pyramid_kernels) * n_pyramid_kernels)
        self.pyramid_pool = PyramidMaxPool(in_channels=512, pyramid_kernel_dims=self.pyramid_kernel_dims)

        # PART-2: The Decoder network
        self.decoder_conv_blk_5 = DecoderBlock3x3(n_layers=3, in_channels=pyramid_out_channels, out_channels=512)  # Output shape: (512, 7, 7)
        self.decoder_conv_blk_4 = DecoderBlock3x3(n_layers=3, in_channels=512, out_channels=256)                   # Output shape: (512, 14, 14)
        self.decoder_conv_blk_3 = DecoderBlock3x3(n_layers=2, in_channels=256, out_channels=128)                   # Output shape: (256, 28, 28)
        self.decoder_conv_blk_2 = DecoderBlock3x3(n_layers=2, in_channels=128, out_channels=64)                    # Output shape: (128, 56, 56)
        self.decoder_conv_blk_1 = DecoderBlock3x3(n_layers=2, in_channels=64, out_channels=n_classes)              # Output shape: (64, 112, 112)

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        # Encoding part
        y_enc_1 = self.encoder_conv_blk_1(X)
        y_enc_2 = self.encoder_conv_blk_2(y_enc_1)
        y_enc_3 = self.encoder_conv_blk_3(y_enc_2)
        y_enc_4 = self.encoder_conv_blk_4(y_enc_3)
        y_enc_5 = self.encoder_conv_blk_5(y_enc_4)

        y_pyramid_pool = self.pyramid_pool(y_enc_5)

        # Decoding part
        # Each decoder block 'i' has skip connection with input of encoder block 'i' (or output of encoder block 'i-1')
        y_dec_5 = self.decoder_conv_blk_5(y_pyramid_pool) + y_enc_4
        y_dec_4 = self.decoder_conv_blk_4(y_dec_5) + y_enc_3
        y_dec_3 = self.decoder_conv_blk_3(y_dec_4) + y_enc_2
        y_dec_2 = self.decoder_conv_blk_2(y_dec_3) + y_enc_1
        y_dec_1 = self.decoder_conv_blk_1(y_dec_2)

        # Softmax layer is not required because nn.Softmax expects raw un-normalized scores
        return y_dec_1
