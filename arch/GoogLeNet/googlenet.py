# inception_v2.py -- Module containing the class for the GoogLeNet model


import torch
import torch.nn as nn


class Inception(nn.Module):
    """
    Class implementing the (non-naive) Inception module with 1x1 convolutions

    in_channels:   Number of channels in the input
    n_axa:         Number of filters for (a x a) kernel sizes
    n_axa_reduce:  Number of filters for (1 x 1) convolution before applying (a x a) convolution
    n_pool_proj:   Number of filters for (1 x 1) convolution after applying the max-pool operation
    """
    def __init__(self, in_channels, n_1x1, n_3x3_reduce, n_3x3, n_5x5_reduce, n_5x5, n_pool_proj):
        super().__init__()

        self.conv_3x3_reduce = nn.Conv2d(in_channels=in_channels, out_channels=n_3x3_reduce, kernel_size=(1, 1))
        self.conv_5x5_reduce = nn.Conv2d(in_channels=in_channels, out_channels=n_5x5_reduce, kernel_size=(1, 1))
        self.pool_proj = nn.Conv2d(in_channels=in_channels, out_channels=n_pool_proj, kernel_size=(1, 1))

        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=n_1x1, kernel_size=(1, 1))
        self.conv_3x3 = nn.Conv2d(in_channels=n_3x3_reduce, out_channels=n_3x3, kernel_size=(3, 3), padding='same')
        self.conv_5x5 = nn.Conv2d(in_channels=n_5x5_reduce, out_channels=n_5x5, kernel_size=(5, 5), padding='same')

        # NOTE: Need the output dimensions to match when doing max-pooling
        #       with the input dimensions. This can be achieved by adding a padding of (kernel_size // 2)
        #       Also note that the stride must be (1, 1) for this to happen, or else the output dims will be halved
        #       if used the default stride of (2, 2)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """

        # It is important to note that the outputs of the inception module only differ in the channel dimension
        # Need to concatenate along the channel dimension to produce the required output
        y_1x1 = self.conv_1x1(X)                           # Output shape: (batch, n_1x1, height, width)
        y_3x3 = self.conv_3x3( self.conv_3x3_reduce(X) )   # Output shape: (batch, n_3x3, height, width)
        y_5x5 = self.conv_5x5( self.conv_5x5_reduce(X) )   # Output shape: (batch, n_5x5, height, width)
        y_maxpool = self.pool_proj( self.max_pool(X) )     # Output shape: (batch, pool_proj, height, width)

        # Concatenate all the outputs along the channel dimension
        # Doesn't matter in the order they are concatenated
        y_inception = torch.cat([y_1x1, y_3x3, y_5x5, y_maxpool], dim=1)
        return y_inception


class GoogLeNet(nn.Module):
    """
    Module for the GoogLeNet model on ImageNette (320px) dataset (ImageNet is too big: ~155GB !! )
    Paper: https://arxiv.org/abs/1409.4842

    NOTE: - The original ImageNet dataset has 1000 classes. ImageNette-320, which is used here,
            has 10 classes. So output layer will have only 10 units, instead of 1000.
          - The image size here (224 x 224) is different from AlexNet (227 x 227)
    """

    n_classes = 10      # The number of classes in the dataset that is being used

    def __init__(self):
        super().__init__()

        # NOTE: 1. The output of first convolution in the paper mentions the output to be (96, 112, 112)
        #          This can only be achieved when the padding is set to (3, 3). Similar reasoning for any
        #          other layers that uses explicit padding -- Trying to match the dimensions in the paper
        #
        #       2. The paper doesn't mention about parameters related to local response normalization
        #          So I'm assuming it is the same as the one that was used in AlexNet, i.e. (n // 2) filters on both
        #          sides of the current filter in consideration.

        self.conv_layers = [                                                  # Input shape:  (3, 224, 224)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7),
                      stride=(2, 2), padding=(3, 3)),                          # Output shape: (64, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),   # Output shape: (64, 56, 56)
            nn.LocalResponseNorm(size=32),

            # The layer below uses one 1x1 convolution that reduces the channel dimension to 64 before doing a
            # (3 x 3) convolution. But the channel dimension is already 64, I guess it is to add one more non-linearity
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),    # Output shape: (64, 56, 56)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3)),   # Output shape: (192, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),   # Output shape: (192, 28, 28)
            nn.LocalResponseNorm(size=96),

            # Inception (3a)
            Inception(in_channels=192, n_1x1=64, n_3x3_reduce=96, n_3x3=128,
                      n_5x5_reduce=16, n_5x5=32, n_pool_proj=32),              # Output shape: (256, 28, 28)
            nn.ReLU(),

            # Inception (3b)
            Inception(in_channels=256, n_1x1=128, n_3x3_reduce=128, n_3x3=192,
                      n_5x5_reduce=32, n_5x5=96, n_pool_proj=64),              # Output shape: (480, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),   # Output shape: (480, 14, 14)

            # Inception (4a)
            Inception(in_channels=480, n_1x1=192, n_3x3_reduce=96, n_3x3=208,
                      n_5x5_reduce=16, n_5x5=48, n_pool_proj=64),              # Output shape: (512, 14, 14)
            nn.ReLU(),

            # Inception (4b)
            Inception(in_channels=512, n_1x1=160, n_3x3_reduce=112, n_3x3=224,
                      n_5x5_reduce=24, n_5x5=64, n_pool_proj=64),              # Output shape: (512, 14, 14)
            nn.ReLU(),

            # Inception (4c)
            Inception(in_channels=512, n_1x1=128, n_3x3_reduce=128, n_3x3=256,
                      n_5x5_reduce=24, n_5x5=64, n_pool_proj=64),              # Output shape: (512, 14, 14)
            nn.ReLU(),

            # Inception (4d)
            Inception(in_channels=512, n_1x1=112, n_3x3_reduce=144, n_3x3=288,
                      n_5x5_reduce=32, n_5x5=64, n_pool_proj=64),              # Output shape: (528, 14, 14)
            nn.ReLU(),

            # Inception (4e)
            Inception(in_channels=528, n_1x1=256, n_3x3_reduce=160, n_3x3=320,
                      n_5x5_reduce=32, n_5x5=128, n_pool_proj=128),            # Output shape: (832, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),   # Output shape: (832, 7, 7)

            # Inception (5a)
            Inception(in_channels=832, n_1x1=256, n_3x3_reduce=160, n_3x3=320,
                      n_5x5_reduce=32, n_5x5=128, n_pool_proj=128),            # Output shape: (832, 7, 7)
            nn.ReLU(),

            # Inception (5b)
            Inception(in_channels=832, n_1x1=384, n_3x3_reduce=192, n_3x3=384,
                      n_5x5_reduce=48, n_5x5=128, n_pool_proj=128),            # Output shape: (1024, 7, 7)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1)),                   # Output shape: (1024, 1, 1)
            nn.Dropout(p=0.4)
        ]

        # Each sample is of shape (1024*1*1, ) = (1024, )
        self.fc_layers = [
            nn.Linear(1024, self.n_classes)
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
