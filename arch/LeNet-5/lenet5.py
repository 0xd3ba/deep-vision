# lenet5.py -- Module containing the class for the LeNet-5 model

import torch.nn as nn


class LeNet5(nn.Module):
    """
    Module for the LeNet-5 model for MNIST dataset
    Paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

    NOTE: In Torch's MNIST, each sample is of shape (1, 28, 28)
          In the original paper, the samples are of shape (1, 32, 32).

          Need to take care of this discrepancy here, by introducing padding
          in the first convolution layer such that the output is the same
          after passing through this layer
    """

    def __init__(self, ip_dim, op_dim):
        super().__init__()

        self.conv_layers_list = [
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding='same'),   # Output: (6, 28, 28)
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),                                # Output: (6, 14, 14)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),                  # Output: (16, 10, 10)
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),                                # Output: (16, 5, 5)
        ]

        # Each sample is of shape (16*5*5, ) = (400, )
        self.feed_forward_list = [
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        ]

        self.conv_model = nn.Sequential(*self.conv_layers_list)
        self.feed_forward_model = nn.Sequential(*self.feed_forward_list)

    def forward(self, X):
        """
        X: (batch_size, 1, 28, 28)

        Returns the scores of the batch, by passing through the convolutions
        and the feed-forward network
        """
        batch_size = X.shape[0]

        conv_y = self.conv_model(X)
        conv_y = conv_y.reshape(batch_size, -1)

        pred_y = self.feed_forward_model(conv_y)
        return pred_y
