# generator.py -- The generator model for Pix2Pix model

import torch
import torch.nn as nn


class DownsamplingBlock(nn.Module):
    """
    Downsampling block for the generator that follows U-Net style architecture:
        - Two 3x3 convolutions
        - Max-pool (output dimensions is exactly the half of input dimensions)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_blk = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, X):
        return self.conv_blk(X)


class UpsamplingBlock(nn.Module):
    """
    Upsampling block for the generator that follows U-Net style architecture:
        - Two 3x3 convolutions
        - Upsampling Layer (output dimensions is exactly twice of input dimensions)

    NOTE: The number of channels are (in_channels + out_channels), because of skip-connect concantenation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=in_channels,
                                           kernel_size=(2, 2), stride=(2, 2))
        self.conv_blk = nn.Sequential(
            nn.Conv2d(in_channels=in_channels+out_channels, out_channels=in_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, X, skip_connect_X):
        """
        X:               (batch, channels_1, height, width)
        skip_connect_X:  (batch, channels_2, 2*height, 2*width)
        """
        y_ups = self.upsample(X)
        y_cat = torch.cat([skip_connect_X, y_ups], dim=1)    # Concatenate on the channel-axis
        y_conv = self.conv_blk(y_cat)

        return y_conv


class Generator(nn.Module):
    """ Generator for Pix2Pix """

    def __init__(self):
        super().__init__()

        # Input image: (1, 128, 128)
        self.encoder_1 = DownsamplingBlock(in_channels=1, out_channels=32)     # Output: (32, 64, 64)
        self.encoder_2 = DownsamplingBlock(in_channels=32, out_channels=64)    # Output: (64, 32, 32)
        self.encoder_3 = DownsamplingBlock(in_channels=64, out_channels=128)   # Output: (128, 16, 16)
        self.encoder_4 = DownsamplingBlock(in_channels=128, out_channels=256)  # Output: (256, 8, 8)

        self.decoder_4 = UpsamplingBlock(in_channels=256, out_channels=128)    # Output: (128, 16, 16)
        self.decoder_3 = UpsamplingBlock(in_channels=128, out_channels=64)     # Output: (64, 32, 32)
        self.decoder_2 = UpsamplingBlock(in_channels=64, out_channels=32)      # Output: (32, 64, 64)
        self.decoder_1 = UpsamplingBlock(in_channels=32, out_channels=1)       # Output: (1, 128, 128)

        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1)),
            nn.Sigmoid()    # Need this because input pixels are normalized to be in range [0, 1]
        )

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y_enc_1 = self.encoder_1(X)
        y_enc_2 = self.encoder_2(y_enc_1)
        y_enc_3 = self.encoder_3(y_enc_2)
        y_enc_4 = self.encoder_4(y_enc_3)

        y_dec_4 = self.decoder_4(y_enc_4, y_enc_3)
        y_dec_3 = self.decoder_3(y_dec_4, y_enc_2)
        y_dec_2 = self.decoder_2(y_dec_3, y_enc_1)
        y_dec_1 = self.decoder_1(y_dec_2, X)

        y_final = self.final_layer(y_dec_1)

        return y_final
