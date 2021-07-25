# voc2012.py -- Dataset class for Pascal VOC 2012 dataset
# NOTE: I'm using an extremely small dataset (only 3 images) for testing purposes
#       Dataset Link: https://www.kaggle.com/huanghanchina/pascal-voc-2012

import math
import pathlib
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.transforms import CenterCrop
from torchvision.transforms import Resize
from torchvision.io import read_image


def closest_power_of_2(d):
    """ Returns the value that is closest to the power of 2 """
    exponent = math.floor(math.log(d, 2))
    return 2**exponent

class VOC2012(Dataset):
    """
    Dataset class for VOC-2012 dataset

    NOTE: There is a need to bring down the input shape to a power of 2
          or it leads to extremely frustrating dimension mismatch between encoder-decoder
          as well as with the ground-truth targets. Having it bring to power of 2 avoids this issue -- The
          paper also uses images, whose both dimensions are powers of 2 respectively
    """

    # There are a total of 20 classes in the actual dataset
    # (I'm not sure if background is also considered one) -- If yes, then make the classes to 21.
    # The samples that I have included, has the largest encoded class as 19, so can't be sure

    # The samples that I have included does not include each and every class
    # NOTE: I have built myself a small annotation csv file for the dataset, to make things easier
    n_classes = 20

    def __init__(self, data_dir, annotation_csv='annotation.csv'):
        self.data_dir = pathlib.Path(data_dir)
        self.annot_path = self.data_dir / annotation_csv

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
        segmented_img_tensor = read_image(segmented_img_path)

        # Resize the original image tensor to prevent encoder/decoder mismtach on every possible scenario
        # The way to avoid dimension mismatch is to ensure that height and width are powers of 2
        h, w = orig_img_tensor.shape[-2:]   # Last two axis represent height and width

        h_new = closest_power_of_2(h)
        w_new = closest_power_of_2(w)

        crop_tfm = CenterCrop(size=(h_new, w_new))  # Crop from the center
        orig_img_tensor = crop_tfm(orig_img_tensor).unsqueeze(0)
        segmented_img_tensor = crop_tfm(segmented_img_tensor)

        # NOTE: The segmented image has a white border, which has a value of 255
        # Remove it by replacing its value with 0. It is needed because the total classes
        # Moreover, the target need to be a Long tensor or else Cross-Entropy loss will throw an error
        segmented_img_tensor = segmented_img_tensor.long()
        segmented_img_tensor[segmented_img_tensor == 255] = 0

        # Now resize to the three scales as mentioned in the paper
        resize_2x = Resize(size=(h_new // 2, w_new // 2))
        resize_4x = Resize(size=(h_new // 4, w_new // 4))
        resize_8x = Resize(size=(h_new // 8, w_new // 8))
        resize_16x = Resize(size=(h_new // 16, w_new // 16))

        orig_img_tensor_downscaled_2x = resize_2x(orig_img_tensor)  # Need batch dimension or errors occur
        orig_img_tensor_downscaled_4x = resize_4x(orig_img_tensor)  # Need batch dimension or errors occur

        segmented_img_tensor_downscaled_4x = resize_4x(segmented_img_tensor)
        segmented_img_tensor_downscaled_8x = resize_8x(segmented_img_tensor)
        segmented_img_tensor_downscaled_16x = resize_16x(segmented_img_tensor)

        return ((orig_img_tensor_downscaled_4x, segmented_img_tensor_downscaled_16x),
                (orig_img_tensor_downscaled_2x, segmented_img_tensor_downscaled_8x),
                (orig_img_tensor, segmented_img_tensor_downscaled_4x))

