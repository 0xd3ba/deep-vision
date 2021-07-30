# loss.py -- Module containing the Loss function for WGAN
# What the critic (or critic) tries to do is, maximize the scores of real images and
# minimize the score of fake images
#
# What the generator tries to do is, maximize the score of fake images

import torch
import torch.nn as nn


class WGANLossFunction(nn.Module):
    """ Loss function for WGAN's both critic network and generator network """
    def __init__(self):
        super().__init__()

    def forward(self, fake_pred, real_pred=None):
        # Because the objective of critic is to MAXIMIZE the distance between the two, in order
        # to convert to a minimization problem, take the negative of the loss. Needed because torch's optimizers
        # minimize the loss
        if real_pred is not None:
            loss = torch.mean(real_pred) - torch.mean(fake_pred)
        else:
            loss = torch.mean(fake_pred)

        loss = -loss

        return loss
