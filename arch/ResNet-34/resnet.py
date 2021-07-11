# resnet.py -- Module containing the class for the 34-layer ResNet-34 model
#              Similar to VGG-16 (Builds upon it to be precise)
#
# PS: Not the most efficient way to write the code for the blocks inside ResNet-34 class, but couldn't
#     think of any other better way :(


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolution Block containing a series of 'n' convolutions along with Batch Normalization
    Number of kernel maps to use depends on the supplied parameter

    NOTE: Since ResNet-34 builds upon VGG-16, I presume that the result of convolution is the same
          as the input size, i.e. how it was previously being done in VGG-16
    """

    def __init__(self, in_channels, out_channels, n_layers=2, kernel_size=(3, 3), initial_stride=(1, 1)):
        super().__init__()
        self.n_kernels = n_layers
        self.ip_channels = in_channels
        self.op_channels = out_channels

        self.conv_layers = []
        curr_channel = in_channels

        # Add the convolution layers
        # NOTE: In the blocks (starting from 2nd block), the initial stride is (2, 2), i.e.
        #       decreases the input size by a factor of 2. This is only done for the
        #       first convolution layer of the block. The remaining strides for the layers
        #       in the block are (1, 1)

        curr_stride = initial_stride
        for i in range(n_layers):

            # We dont want output of convolution operation to be the same as the input
            # size when we want a strided-convolution. Need a padding of (1, 1) to ensure dimensions
            # match with the 1x1 convolution of the residual connection (with stride (2,2))
            if curr_stride != (1, 1):
                padding = (1, 1)
            else:
                padding = 'same'

            conv_layer_i = nn.Conv2d(in_channels=curr_channel,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=curr_stride,
                                     padding=padding)
            curr_channel = out_channels
            curr_stride = (1, 1)
            self.conv_layers.append(conv_layer_i)
            self.conv_layers.append(nn.BatchNorm2d(out_channels))   # Batch Normalization (not present in VGG-16)
            self.conv_layers.append(nn.ReLU())

        # NOTE: Unlike VGG-16, there is no max-pool layer after the convolutions

        # Build the block as a model
        self.conv_block = nn.Sequential(*self.conv_layers)

    def forward(self, X):
        """
        X:  (batch, n_channels, height, width)
        """
        return self.conv_block(X)


class ResNet(nn.Module):
    """
    Module for the ResNet-34 model on ImageNette (320px) dataset (ImageNet is too big: ~155GB !! )
    Paper: https://arxiv.org/abs/1512.03385

    NOTE: - The original ImageNet dataset has 1000 classes. ImageNette-320, which is used here,
            has 10 classes. So output layer will have only 10 units, instead of 1000.
    """

    n_classes = 10      # The number of classes in the dataset that is being used

    def __init__(self):
        super().__init__()

        # Build the convolution blocks as mentioned in the paper
        # Residual connections are two convolution blocks apart, i.e. there is a connection from input of block 'i'
        # to input of block 'i+2'

        # Initial convolution block
        # Input size:   (3, 224, 224)
        # Output size:  (64, 110, 110) -- pooling --> (64, 54, 54)
        self.init_conv_blk = ConvBlock(n_layers=1, in_channels=3, out_channels=64, kernel_size=(7, 7), initial_stride=(2, 2))
        self.max_pool_lyr = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        # Block-1 (6 convolution layers)
        # Input size:  (64, 54, 54)
        # Output size: (64, 54, 54)
        self.conv_1_p1 = ConvBlock(in_channels=64, out_channels=64)
        self.conv_1_p2 = ConvBlock(in_channels=64, out_channels=64)
        self.conv_1_p3 = ConvBlock(in_channels=64, out_channels=64)

        # Block-2 (8 convolution layers)
        # Input size:  (64, 54, 54)
        # Output size: (128, 27, 27)
        self.conv_2_tfm = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))  # Transform for residual connection
        self.conv_2_p1 = ConvBlock(in_channels=64, out_channels=128, initial_stride=(2, 2))
        self.conv_2_p2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_2_p3 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_2_p4 = ConvBlock(in_channels=128, out_channels=128)

        # Block-3 (12 convolution layers)
        # Input size:  (128, 27, 27)
        # Output size: (256, 14, 14)
        self.conv_3_tfm = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))  # Transform for residual connection
        self.conv_3_p1 = ConvBlock(in_channels=128, out_channels=256, initial_stride=(2, 2))
        self.conv_3_p2 = ConvBlock(in_channels=256, out_channels=256)
        self.conv_3_p3 = ConvBlock(in_channels=256, out_channels=256)
        self.conv_3_p4 = ConvBlock(in_channels=256, out_channels=256)
        self.conv_3_p5 = ConvBlock(in_channels=256, out_channels=256)
        self.conv_3_p6 = ConvBlock(in_channels=256, out_channels=256)

        # Block-4 (6 convolution layers)
        # Input size:  (256, 14, 14)
        # Output size: (512, 7, 7)
        self.conv_4_tfm = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))  # Transform for residual connection
        self.conv_4_p1 = ConvBlock(in_channels=256, out_channels=512, initial_stride=(2, 2))
        self.conv_4_p2 = ConvBlock(in_channels=512, out_channels=512)
        self.conv_4_p3 = ConvBlock(in_channels=512, out_channels=512)

        # Average pooling layer
        # No mention about kernel size and stride in the paper, so assuming they are the same as the
        # max-pooling layer, i.e. kernel size is (3, 3) with stride (2, 2)
        # Input size:  (512, 7, 7)
        # Output size: (512, 3, 3)
        self.avg_pool_lyr = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))

        # Input to the fully-connected network is (512*3*3, ) = (4608, )
        # There is only a single layer before the output, i.e. no hidden layers in between
        self.fc_layers = [
            nn.Linear(4608, self.n_classes)
        ]

        # Build the fully connected model
        self.fc_model = nn.Sequential(*self.fc_layers)

    def forward(self, X):
        """
        X:  (batch, 3, 224, 224)
        """
        batch_size = X.shape[0]

        y = self.init_conv_blk(X)                       # Shape: (batch_size, 64, 110, 110)
        y = self.max_pool_lyr(y)                        # Shape: (batch_size, 64, 54, 54)

        y = self.conv_1_p1(y) + y                       # Shape: (batch_size, 64, 54, 54)
        y = self.conv_1_p2(y) + y
        y = self.conv_1_p3(y) + y

        y = self.conv_2_p1(y) + self.conv_2_tfm(y)      # Shape: (batch_size, 128, 27, 27)
        y = self.conv_2_p2(y) + y
        y = self.conv_2_p3(y) + y

        y = self.conv_3_p1(y) + self.conv_3_tfm(y)      # Shape: (batch_size, 256, 14, 14)
        y = self.conv_3_p2(y) + y
        y = self.conv_3_p3(y) + y
        y = self.conv_3_p4(y) + y
        y = self.conv_3_p5(y) + y
        y = self.conv_3_p6(y) + y

        y = self.conv_4_p1(y) + self.conv_4_tfm(y)      # Shape: (batch_size, 512, 7, 7)
        y = self.conv_4_p2(y) + y
        y = self.conv_4_p3(y) + y
        y = self.avg_pool_lyr(y)                        # Shape: (batch_size, 512, 3, 3)

        # I know there might be a better way to handle the above mess, but it doesn't matter as I'm
        # not writing production code or anything - just for the sake of learning.

        y = y.reshape(batch_size, -1)       # Flatten the output of the convolutional layers
        y = self.fc_model(y)                # Finally get the raw scores for each class. Shape: (batch, n_classes)

        return y
