# unet.py -- Module containing the class for the UNet model
#
# NOTE: One small difference from the original model that I've done is that
#       the (3,3) convolutions are set to produce output which is same as input shape, i.e. padding is set to 'same'
#       This has been done because I'm using a different dataset, so not doing this causes
#       dimension mismatch.

import torch
import torch.nn as nn


class UNetEncoder(nn.Module):
    """
    Encoder block with two straightforward 3x3 convolutions followed
    by a 2x2 max-pool with stride of 2
    """
    def __init__(self, in_channels, out_channels, has_max_pool=True):
        super().__init__()
        self.has_max_pool = has_max_pool
        self.conv_layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        ]

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.encoder_block = nn.Sequential(*self.conv_layers)

    def forward(self, X):
        y_enc = self.encoder_block(X)
        y_pool = y_enc
        if self.has_max_pool:
            y_pool = self.max_pool(y_pool)

        # Returning two of them, because the output of convolution block (before pooling, if used)
        # is used for skip-connection
        return y_enc, y_pool



class UNetDecoder(nn.Module):
    """ Decoder block with one tranposed convolution (to undo the max-pool),
    followed by two straight-forward 3x3 convolutions """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=(2, 2), stride=(2, 2))

        # Note that the input to first convolution operation has output of encoder block concatenated at
        # channel dimension, hence the number of channels is double
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )

    def _crop_encoder_output(self, x_enc, x_dec):
        """
        Crops the encoder output to match the shape of decoder's output
        Assumption is that the encoder's output is greater than decoder's output
        (which will be, as the paper uses some extra convolutions without downsampling at the bottom)
        """
        enc_height, enc_width = x_enc.shape[-2], x_dec.shape[-1]

        height_diff = enc_height - x_dec.shape[-2]
        width_diff = enc_width - x_dec.shape[-1]

        skip_height = height_diff // 2      # How many pixels to skip at top and bottom
        skip_width = width_diff // 2        # How many pixels to skip at left and right

        cropped_x_enc = x_enc[:, :, skip_height:(enc_height - skip_height), skip_width:(enc_width - skip_width)]
        return cropped_x_enc

    def forward(self, X, skip_connect_X):
        """
        X:              Input to the decoder
        skip_connect_X: The data that needs to be concatenated after tranposed convolution

        NOTE: The dimension of the inputs mismatch, in particular, dimensions of X > dimensions of skip_connect_X.
              Since the paper mentions to crop the boundaries, this is what will be done to X match the dimensions.
        """
        y_conv_t = self.conv_transpose(X)
        enc_cropped = self._crop_encoder_output(x_enc=skip_connect_X, x_dec=y_conv_t)

        # Concatenate both on channel dimensions: (batch, channel, height, width)
        # Then apply the convolution
        y_conv = torch.cat([y_conv_t, enc_cropped], dim=1)
        y_conv = self.conv_block(y_conv)
        return y_conv


class UNet(nn.Module):
    """
    Class for the UNet model
    """

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes      # The number of channels in the final decoder output

        # NOTE: I'm assuming (572 x 572) RGB image (3 channels) as input for simplicity
        #       Note that this can be applied to any (sufficiently large) image, i.e. it is not just limited
        #       to the above mentioned dimension, similar to FCN and SegNet

        self.encoder_blk_1 = UNetEncoder(in_channels=3, out_channels=64)
        self.encoder_blk_2 = UNetEncoder(in_channels=64, out_channels=128)
        self.encoder_blk_3 = UNetEncoder(in_channels=128, out_channels=256)
        self.encoder_blk_4 = UNetEncoder(in_channels=256, out_channels=512)
        self.encoder_blk_5 = UNetEncoder(in_channels=512, out_channels=1024, has_max_pool=False)

        # Now comes the decoder.
        # Naming them in reverse order to understand which decoder block relates to which encoder block
        self.decoder_blk_4 = UNetDecoder(in_channels=1024, out_channels=512)
        self.decoder_blk_3 = UNetDecoder(in_channels=512, out_channels=256)
        self.decoder_blk_2 = UNetDecoder(in_channels=256, out_channels=128)
        self.decoder_blk_1 = UNetDecoder(in_channels=128, out_channels=64)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=(1, 1))

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y_enc_1, y_pool_1 = self.encoder_blk_1(X)
        y_enc_2, y_pool_2 = self.encoder_blk_2(y_pool_1)
        y_enc_3, y_pool_3 = self.encoder_blk_3(y_pool_2)
        y_enc_4, y_pool_4 = self.encoder_blk_4(y_pool_3)
        y_enc_5, y_pool_5 = self.encoder_blk_5(y_pool_4)    # This doesn't have max-pooling, so y_enc_5 = y_pool_5

        y_dec_4 = self.decoder_blk_4(y_pool_5, skip_connect_X=y_enc_4)
        y_dec_3 = self.decoder_blk_3(y_dec_4, skip_connect_X=y_enc_3)
        y_dec_2 = self.decoder_blk_2(y_dec_3, skip_connect_X=y_enc_2)
        y_dec_1 = self.decoder_blk_1(y_dec_2, skip_connect_X=y_enc_1)

        y_final  = self.final_conv(y_dec_1)
        return y_final
