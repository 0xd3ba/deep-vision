# icnet.py --  Module containing the class for the ICNet model
#              Uses a UNet style Encoder network, with dilated convolutions for layer-1


import torch
import torch.nn as nn
from backbone import EncoderNetwork
from cff import CFF


class ICNet(nn.Module):
    """
    Class for the ICNet model
    Paper: https://arxiv.org/abs/1704.08545
    """

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        # The most-intensive convolutional block, which takes as input, the smallest image
        # i.e. downsampled by a factor of 4
        # NOTE: The final number of output channels is half the number of dilation channels (as it
        #       was not mentioned in the paper, so I assumed this)
        self.enc_layer_s = EncoderNetwork(channels_list=[128, 256, 512],
                                          use_dilations=True,
                                          dilation_rates=[2, 4, 6],
                                          dilation_op_channels=[256, 512, 1024])

        self.enc_layer_m = EncoderNetwork(channels_list=[64, 128, 256])     # Takes as input, the medium-sized image
        self.enc_layer_l = EncoderNetwork(channels_list=[32, 64, 128])      # Takes as input, the full-sized image

        # Build the CFF layers
        # Note that output of CFF_12 is fed to CFF_23 as f1, f2 being the output of enc_layer_l
        self.cff_12 = CFF(n_classes=n_classes, f1_in_channels=1024//2, f2_in_channels=256, out_channels=256)  # CFF between layer 1 and 2
        self.cff_23 = CFF(n_classes=n_classes, f1_in_channels=256, f2_in_channels=128, out_channels=128)      # CFF between layer 2 and 3

        # The paper doesn't mention this, but we need a separate 1x1 convolution block for output of cff_23
        # or else it will not be possible to train with the corresponding ground-truth target
        self.layer_3_upsampler_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.cff_23_proj = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=(1, 1))

    def forward(self, x1, x2, x3):
        """
        x1: (batch, 3, height/4, width/4)    -- Original image scaled down by a factor of 4
        x2: (batch, 3, height/2, width/2)    -- Original image scaled down by a factor of 2
        x3: (batch, 3, height, width)        -- Original image with no scaling
        """
        y1 = self.enc_layer_s(x1)
        y2 = self.enc_layer_m(x2)
        y3 = self.enc_layer_l(x3)

        # Now feed them to the cascade-feature-fusion units
        f1_pred_scores, y_cff_12 = self.cff_12(f1=y1, f2=y2)
        f2_pred_scores, y_cff_23 = self.cff_23(f1=y_cff_12, f2=y3)

        y3_ups_scores = self.layer_3_upsampler_1(y_cff_23)
        y3_ups_scores = self.cff_23_proj(y3_ups_scores)

        return f1_pred_scores, f2_pred_scores, y3_ups_scores