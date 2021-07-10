# zfnet.py -- Module containing the class for the ZFNet-16 model


import torch
import torch.nn as nn


class ZFNet(nn.Module):
    """
    Module for the ZFNet model on ImageNette (320px) dataset (ImageNet is too big: ~155GB !! )
    Paper: https://arxiv.org/abs/1311.2901

    NOTE: - The original ImageNet dataset has 1000 classes. ImageNette-320, which is used here,
            has 10 classes. So output layer will have only 10 units, instead of 1000.
          - The model is same as AlexNet, but with few differences in the kernel sizes and strides
          - The image size here (224 x 224) is different from AlexNet (227 x 227)
    """

    n_classes = 10      # The number of classes in the dataset that is being used

    def __init__(self):
        super().__init__()

        # NOTE: The output of first convolution in the paper mentions the output to be (96, 110, 110)
        #       If convolutions are used as is, the result comes out to be (96, 109, 109). Need to add
        #       1 pixel padding to ensure the output comes out to be 110 instead of 109. Similar reasoning
        #       for other layers, wherever padding is used.

        self.conv_layers = [                                                         # Input size:  (3, 224, 224)
            nn.Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1)),     # Output size: (96, 110, 110)
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),         # Output size: (96, 55, 55)
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2)),                   # Output size: (256, 26, 26)
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),         # Output size: (256, 13, 13)

            # Rest are similar to AlexNet
            # Now comes convolutions without any max-pooling
            # The shape remains the same however
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding='same'),  # Output size: (384, 13, 13)
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding='same'),  # Output size: (384, 13, 13)
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding='same'),  # Output size: (256, 13, 13)

            # The final max-pool layer
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))           # Output size: (256, 6, 6)
        ]

        # Each sample is of shape (256*6*6, ) = (9216, )
        self.fc_layers = [
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # AlexNet uses dropout with probability 0.5
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.n_classes)
        ]

        self.conv_net = nn.Sequential(*self.conv_layers)
        self.fc_net = nn.Sequential(*self.fc_layers)

    def forward(self, X):
        """
        X: (batch_size, 3, 227, 227)
        """
        batch_size = X.shape[0]
        y = self.conv_net(X)
        y = y.reshape(batch_size, -1)  # Reshape to match input size of fully connected layers
        y = self.fc_net(y)

        # Not using Softmax activation because Cross-Entropy loss requires un-normalized scores
        return y
