# voc2012.py -- Dataset class for Pascal VOC 2012 dataset
# NOTE: I'm using an extremely small dataset (only 3 images) for testing purposes
#       Dataset Link: https://www.kaggle.com/huanghanchina/pascal-voc-2012

import pathlib
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.transforms import CenterCrop
from torchvision.io import read_image


class VOC2012(Dataset):
    """
    Dataset class for VOC-2012 dataset

    If we have 'n' downsampling layers such that the downsampling reduces the size by a factor of 2 everytime,
    then we need to ensure that the height and width of the image are multiples of 2^n in order to avoid dimension
    mismatch between encoder and decoder outputs
    """

    # There are a total of 20 classes in the actual dataset
    # (I'm not sure if background is also considered one) -- If yes, then make the classes to 21.
    # The samples that I have included, has the largest encoded class as 19, so can't be sure

    # The samples that I have included does not include each and every class
    # NOTE: I have built myself a small annotation csv file for the dataset, to make things easier
    n_classes = 20

    def __init__(self, data_dir, annotation_csv='annotation.csv', n_downsampling=5):
        self.data_dir = pathlib.Path(data_dir)
        self.annot_path = self.data_dir / annotation_csv
        self.downsample_factor = int(2**n_downsampling)

        # The names of the columns of the CSV file storing the path
        # to the original image and the segmented image
        self.orig_col = 'orig_path'
        self.segmented_col = 'segmented_path'

        self.annot_df = pd.read_csv(self.annot_path)

    def __len__(self):
        return len(self.annot_df)

    def __getitem__(self, i):
        """
        Returns the i'th element in the dataframe

        WARNING: Make sure that the index is a valid index, i.e. it must not be
        greater than the size of the dataframe. Ensure this by building the dataloader
        appropriately !
        """
        orig_img_path = str(self.data_dir / self.annot_df[self.orig_col][i])
        segmented_img_path = str(self.data_dir / self.annot_df[self.segmented_col][i])

        # Load the images and normalize the training image
        orig_img_tensor = read_image(orig_img_path) / 255.0
        segmented_img_tensor = read_image(segmented_img_path)[0]  # Has only one channel, so remove it

        # Resize the original image tensor to prevent encoder/decoder mismtach on every possible scenario
        # The way to avoid dimension mismatch is to ensure that height and width are multiples of the
        # down-sampling factor.
        h, w = orig_img_tensor.shape[-2:]   # Last two axis represent height and width
        h_new = self.downsample_factor * (h // self.downsample_factor)
        w_new = self.downsample_factor * (w // self.downsample_factor)

        crop_tfm = CenterCrop(size=(h_new, w_new))  # Crop from the center
        orig_img_tensor = crop_tfm(orig_img_tensor)
        segmented_img_tensor = crop_tfm(segmented_img_tensor)

        # NOTE: The segmented image has a white border, which has a value of 255
        # Remove it by replacing its value with 0. It is needed because the total classes
        # Moreover, the target need to be a Long tensor or else Cross-Entropy loss will throw an error
        segmented_img_tensor = segmented_img_tensor.long()
        segmented_img_tensor[segmented_img_tensor == 255] = 0

        return orig_img_tensor, segmented_img_tensor

