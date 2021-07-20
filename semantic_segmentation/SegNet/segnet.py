# segnet.py -- Module containing the class for the SegNet model
#              Similar to FCN, that uses VGG-16, but instead of transposed convolution
#              it uses "max unpooling". Everything else is the same as FCN implementation
#

import torch
import torch.nn as nn


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
        pool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)   # Need to return indices
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
    Decoder block that first upsamples the input using Max-Unpooling layer and then does vanilla
    convolution to "dense" out the sparse un-pooled tensor
    """
    def __init__(self, n_layers, in_channels, out_channels):
        super().__init__()
        self.n_kernels = n_layers
        self.ip_channels = in_channels
        self.op_channels = out_channels

        self.conv_layers = []
        curr_channel = in_channels

        # The max-unpooling layer to undo the max-pool
        self.unpool_layer = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))

        # Add the normal convolution layers
        # One good thing about VGG-16 is that all convolutions blocks are consistent across the kernel sizes
        # So, can just do everything in a single loop
        for i in range(n_layers):
            conv_layer_i = nn.Conv2d(in_channels=curr_channel,
                                     out_channels=out_channels,
                                     kernel_size=(3, 3),
                                     padding='same')
            curr_channel = out_channels
            self.conv_layers.append(conv_layer_i)
            self.conv_layers.append(nn.BatchNorm2d(num_features=out_channels))
            self.conv_layers.append(nn.ReLU())

        # Build the block as a model
        self.conv_block = nn.Sequential(*self.conv_layers)

    def forward(self, X, pool_indices):
        """
        X:  (batch, n_channels, height, width)
        """
        y_unpool = self.unpool_layer(X, pool_indices)
        return self.conv_block(y_unpool)


class SegNet(nn.Module):
    """
    Class for the SegNet model

    Because the VGG-16 version is being implemented, the input images muse be atleast
    (224, 224) with 3 channels (RGB). The images can be greater in any dimension, but can't go less than 224

    """

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        # I'm assuming (3, 224, 224) image as calculation of output shapes in tranposed-convolution
        # is non-trivial. Leads to some rounding and stuff, which means it might produce different shape
        # from the input that was used in the convolution operation previously.

        # PART-1: The Encoder network
        self.encoder_conv_blk_1 = EncoderBlock3x3(n_layers=2, in_channels=3, out_channels=64)      # Output shape: (64, 112, 112)
        self.encoder_conv_blk_2 = EncoderBlock3x3(n_layers=2, in_channels=64, out_channels=128)    # Output shape: (128, 56, 56)
        self.encoder_conv_blk_3 = EncoderBlock3x3(n_layers=2, in_channels=128, out_channels=256)   # Output shape: (256, 28, 28)
        self.encoder_conv_blk_4 = EncoderBlock3x3(n_layers=3, in_channels=256, out_channels=512)   # Output shape: (512, 14, 14)
        self.encoder_conv_blk_5 = EncoderBlock3x3(n_layers=3, in_channels=512, out_channels=512)   # Output shape: (512, 7, 7)

        # PART-2: The Decoder network
        self.decoder_conv_blk_5 = DecoderBlock3x3(n_layers=3, in_channels=512, out_channels=512)         # Output shape: (512, 7, 7)
        self.decoder_conv_blk_4 = DecoderBlock3x3(n_layers=3, in_channels=512, out_channels=256)         # Output shape: (512, 14, 14)
        self.decoder_conv_blk_3 = DecoderBlock3x3(n_layers=2, in_channels=256, out_channels=128)         # Output shape: (256, 28, 28)
        self.decoder_conv_blk_2 = DecoderBlock3x3(n_layers=2, in_channels=128, out_channels=64)          # Output shape: (128, 56, 56)
        self.decoder_conv_blk_1 = DecoderBlock3x3(n_layers=2, in_channels=64, out_channels=n_classes)    # Output shape: (64, 112, 112)

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        # Encoding part
        # Note that the max-pooling layer returns the indices of the selected values
        y_enc_1, idx_1 = self.encoder_conv_blk_1(X)
        y_enc_2, idx_2 = self.encoder_conv_blk_2(y_enc_1)
        y_enc_3, idx_3 = self.encoder_conv_blk_3(y_enc_2)
        y_enc_4, idx_4 = self.encoder_conv_blk_4(y_enc_3)
        y_enc_5, idx_5 = self.encoder_conv_blk_5(y_enc_4)

        # Decoding part
        # Each decoder block 'i' has skip connection with input of encoder block 'i' (or output of encoder block 'i-1')
        y_dec_5 = self.decoder_conv_blk_5(y_enc_5, idx_5) + y_enc_4
        y_dec_4 = self.decoder_conv_blk_4(y_dec_5, idx_4) + y_enc_3
        y_dec_3 = self.decoder_conv_blk_3(y_dec_4, idx_3) + y_enc_2
        y_dec_2 = self.decoder_conv_blk_2(y_dec_3, idx_2) + y_enc_1
        y_dec_1 = self.decoder_conv_blk_1(y_dec_2, idx_1)

        # Softmax layer is not required because nn.Softmax expects raw un-normalized scores
        return y_dec_1
