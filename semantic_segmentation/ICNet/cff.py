# cff.py -- Module containing the Casecade Feature Fusion Unit

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFF(nn.Module):
    """
    Module for Cascade Feature Fusion Unit
    """

    def __init__(self, n_classes, f1_in_channels, f2_in_channels, out_channels):
        super().__init__()
        self.f1_in_channels = f1_in_channels
        self.f2_in_channels = f2_in_channels
        self.out_channels = out_channels

        self.f1_upsampler = nn.Upsample(scale_factor=2, mode='bilinear')    # Need to up-sample by a factor of 2

        self.f1_dilated_conv = nn.Conv2d(in_channels=f1_in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(3, 3),
                                         dilation=(2, 2),
                                         padding='same')

        # This will be a branch to robustly train the layer corresponding to f1
        self.f1_predict_scores = nn.Conv2d(in_channels=f1_in_channels, out_channels=n_classes, kernel_size=(1, 1))

        self.f2_projection = nn.Conv2d(in_channels=f2_in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.f1_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.f2_batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, f1, f2):
        """
        f1: (batch, channels_f1, height/2, width/2)
        f2: (batch, channels_f2, height, width)

        f1 is exactly the half-size of f2
        """
        y_f1 = self.f1_upsampler(f1)
        y_f1_scores = self.f1_predict_scores(y_f1)      # Scores are calculated immediately after upsampling

        y_f1 = self.f1_dilated_conv(y_f1)
        y_f1 = self.f1_batch_norm(y_f1)

        y_f2 = self.f2_projection(f2)
        y_f2 = self.f2_batch_norm(y_f2)

        # Now need to add both y_f1 and y_f2 before doing a ReLU
        y_comb = y_f1 + y_f2
        y_comb = F.relu(y_comb)

        # Need to return a tuple because of auxillary branch training
        # Note that this is unused during evaluation
        return y_f1_scores, y_comb
