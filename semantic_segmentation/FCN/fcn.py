# fcn.py -- Module containing the class for the Fully-Convolutional Network model
#           I decided to implement the AlexNet version of this. As for why it was chosen, for simplicity of course.
#           VGG-16 has too much parameters ! GoogLeNet is too complicated (see my implementation on /arch/GoogleNet)

import torch
import torch.nn as nn


class FCN(nn.Module):
    """
    Class for the Fully-Convolutional Network model
    """

    def __init__(self):
        super().__init__()
        # TODO: Implement the model

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        # TODO: Implement the model
        pass
