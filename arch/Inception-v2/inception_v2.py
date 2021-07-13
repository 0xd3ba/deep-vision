# inception_v2.py -- Module containing the class for the Inception-v2 model


import torch
import torch.nn as nn
from inception_blocks.modded_5x5_block import InceptionModded5x5
from inception_blocks.asymmetric_block import InceptionAsymmetric7x7
from inception_blocks.coarse_block import InceptionCoarse


class InceptionV2(nn.Module):
    """
    Module for the Inception-v2 model on ImageNette (320px) dataset (ImageNet is too big: ~155GB !! )
    Paper: https://arxiv.org/abs/1512.00567

    NOTE: - The original ImageNet dataset has 1000 classes. ImageNette-320, which is used here,
            has 10 classes. So output layer will have only 10 units, instead of 1000.
          - The image size here (299 x 299) is different from GoogLeNet (224 x 224)

    IMPORTANT: The parameters to inception blocks are NOT mentioned in the paper. But it is mentioned that an external
               ".txt" file contains it, which cannot be found ofcourse. So the parameters to the inception blocks are
               just my (un-tested) configuration
    """

    n_classes = 10      # The number of classes in the dataset that is being used

    def __init__(self):
        super().__init__()

        self.conv_layers = [                                                                  # Input shape:  (3, 299, 299)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),     # Output shape: (32, 149, 149)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),                   # Output shape: (32, 147, 147)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),   # Output shape: (64, 147, 147)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),                                  # Output shape: (64, 73, 73)
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3, 3)),                   # Output shape: (80, 71, 71)
            nn.ReLU(),
            nn.Conv2d(in_channels=80, out_channels=192, kernel_size=(3, 3), stride=(2, 2)),   # Output shape: (192, 35, 35)
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=288, kernel_size=(3, 3), padding='same'), # Output shape: (288, 35, 35)
            nn.ReLU(),

            # Now comes the new (and improvised) Inception blocks
            # Because they didn't mention the exact configuration of the filter sizes, these
            # are pure speculations of mine, to match the output shape.

            # Part-1: 3x Inception blocks (similar to original Inception but with modified 5x5 convolutions)
            # Given input shape:     (288, 35, 35)
            # Required output shape: (768, 17, 17)
            InceptionModded5x5(in_channels=288, n_1x1=96, n_3x3_reduce=96, n_3x3=192,         # Output shape: (480, 35, 35)
                               n_5x5_reduce=48, n_5x5=96, n_pool_proj=96),
            nn.ReLU(),
            InceptionModded5x5(in_channels=480, n_1x1=96, n_3x3_reduce=96, n_3x3=192,         # Output shape: (576, 35, 35)
                               n_5x5_reduce=96, n_5x5=192, n_pool_proj=96),
            nn.ReLU(),
            InceptionModded5x5(in_channels=576, n_1x1=192, n_3x3_reduce=96, n_3x3=288,        # Output shape: (768, 35, 35)
                               n_5x5_reduce=96, n_5x5=192, n_pool_proj=96),

            # This max-pool layer is not mentioned in the paper, but have to add it to make
            # the output shape to (768, 17, 17)
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),                                  # Output shape: (768, 17, 17)

            # Part-2: 5x (asymmetric) Inception blocks
            # Given input shape:     (768, 17, 317)
            # Required output shape: (1280, 8, 8)
            InceptionAsymmetric7x7(in_channels=768, n_1x1=216, n_7x7_reduce=96, n_7x7=216,    # Output shape: (864, 17, 17)
                                   n_13x13_reduce=96, n_13x13=216, n_pool_proj=216),
            nn.ReLU(),
            InceptionAsymmetric7x7(in_channels=864, n_1x1=184, n_7x7_reduce=96, n_7x7=248,    # Output shape: (864, 17, 17)
                                   n_13x13_reduce=96, n_13x13=248, n_pool_proj=184),
            nn.ReLU(),
            InceptionAsymmetric7x7(in_channels=864, n_1x1=216, n_7x7_reduce=96, n_7x7=264,    # Output shape: (960, 17, 17)
                                   n_13x13_reduce=96, n_13x13=264, n_pool_proj=216),
            nn.ReLU(),
            InceptionAsymmetric7x7(in_channels=960, n_1x1=264, n_7x7_reduce=96, n_7x7=312,    # Output shape: (1056, 17, 17)
                                   n_13x13_reduce=96, n_13x13=216, n_pool_proj=264),
            nn.ReLU(),
            InceptionAsymmetric7x7(in_channels=1056, n_1x1=352, n_7x7_reduce=96, n_7x7=352,    # Output shape: (1280, 17, 17)
                                   n_13x13_reduce=96, n_13x13=288, n_pool_proj=288),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            # Part-3: 3x (coarse) Inception blocks
            # Given input shape:     (1280, 8, 8)
            # Required output shape: (2048, 8, 8)
            InceptionCoarse(in_channels=1280, n_1x1=392, n_3x1_3_reduce=96, n_3x1_3=392,      # Output shape: (1568, 8, 8)
                            n_1x1_3_reduce=1, n_1x1_3=392, n_pool_proj=392),
            nn.ReLU(),
            InceptionCoarse(in_channels=1568, n_1x1=464, n_3x1_3_reduce=96, n_3x1_3=464,      # Output shape: (1856, 8, 8)
                            n_1x1_3_reduce=96, n_1x1_3=464, n_pool_proj=464),
            nn.ReLU(),
            InceptionCoarse(in_channels=1856, n_1x1=512, n_3x1_3_reduce=96, n_3x1_3=512,      # Output shape: (2048, 8, 8)
                            n_1x1_3_reduce=96, n_1x1_3=512, n_pool_proj=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(1, 1))                                   # Output shape: (2048, 1, 1)
        ]

        # Each sample is of shape (2048*1*1, ) = (2048, )
        self.fc_layers = [
            nn.Linear(2048, self.n_classes)
        ]

        self.conv_net = nn.Sequential(*self.conv_layers)
        self.fc_net = nn.Sequential(*self.fc_layers)

    def forward(self, X):
        """
        X: (batch_size, 3, 299, 299)
        """
        batch_size = X.shape[0]
        y = self.conv_net(X)
        y = y.reshape(batch_size, -1)  # Reshape to match input size of fully connected layers
        y = self.fc_net(y)

        # Not using Softmax activation because Cross-Entropy loss requires un-normalized scores
        return y
