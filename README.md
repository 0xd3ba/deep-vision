# Deep Vision
Computer-Vision related research papers implemented using PyTorch, from scratch.

- `/arch/*` contains various CNN architectures
- `/semantic_segmentation/*` contains CNN models related to the task of semantic segmentation
- `/gan/*` contains models for generative adversarial networks

The following list of papers have been currently implemented:

### Architectures
| Model      | Year | Paper |
|------------|------|------|
LeNet-5      | 1998 | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) |
AlexNet      | 2012 | [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
ZFNet        | 2013 | [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) |
VGG-16       | 2014 | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) |
GoogLeNet    | 2014 | [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) |
ResNet       | 2015 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
Inception-v2 | 2015 | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) |
Xception     | 2016 | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) |
MobileNet    | 2017 | [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

### Semantic Segmentation
| Model      | Year | Paper |
|------------|------|------|
FCN          | 2014 | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) |
SegNet       | 2015 | [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561) |
U-Net        | 2015 | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) |
PSPNet       | 2015 | [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) |
ENet         | 2016 | [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147) |
ICNet        | 2017 | [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545) |

### Generative Adversarial Networks (GANs)
| Model      | Year | Paper |
|------------|------|------|
GAN          | 2014 | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) |
DCGAN        | 2015 | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) |
Pix2Pix      | 2016 | [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) |
WGAN         | 2017 | [Wasserstein GAN](https://arxiv.org/abs/1701.07875) |


## Training / Configuring the Models
The only dependencies required are 
- `torch` (`v1.9.0` or similar) 
- `numpy` (`v1.20.1` or similar)
- `tqdm`

**NOTE:** Many of the models use external datasets, some of which have not been pushed to this repository due to the sheer
amount of size. Cross-check `/[model_class]/[model]/dataset/` directory and see the `*.py` file for information about
the dataset that needs to be used. 
Alternatively, you can also use any relevant dataset of your choice, but it's upto you
to write the dataset class and change the required parameters (more on it below).

Follow the steps to train the model:

- Change to the relevant model directory: `$ cd /[model_class]/[model]/`
- Change/Download the dataset, if necessary and place the contents in `dataset/` directory.
- Update `main.py` and/or `train.py` (necessary if using a different dataset)
- Execute `main.py`:  `$ python3 main.py`

**#TODO:** 
- [ ] Add code for saving the models