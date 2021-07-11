# imagenette.py -- Dataset class for the ImageNette-320 dataset

import pathlib
import pandas as pd
import torch
from torchvision.transforms.transforms import Resize
from torchvision.io import read_image
from torch.utils.data.dataloader import Dataset


class ImageNetteDataset(Dataset):
    """
    Dataset class for the ImageNette-320 dataset.
    The annotation file has the following that comes with it has the following columns (in order):

    path    noisy_labels_0    noisy_labels_1    noisy_labels_5    noisy_labels_25    noisy_labels_50    is_valid

    The number in the "noisy_labels_*" represents the fraction of labels that are incorrect in the annotation.
    We only need the correct labels in our case, so we can ignore others and use only "noisy_labels_0", i.e
    the true labels

    Also note that each image file is of shape (3, 320, 320)
    Since ResNet-34 was written for images of shape (3, 224, 224), need to rescale appropriately
    """

    imagenet_dims = (224, 224)

    def __init__(self, data_dir, annotation_csv, train=True):
        self.data_dir = pathlib.Path(data_dir)
        self.annotation_df = pd.read_csv(self.data_dir / annotation_csv)
        self.transform = Resize(size=self.imagenet_dims)
        self.is_validation = not train
        self.images_df = None

        self.path_col = 'path'                  # The name of the column storing the paths
        self.label_col = 'noisy_labels_0'       # The name of the column storing the required labels

        # Create an enumeration mapping label names to integers
        # And an inverse of it, i.e. mapping integers to the label names
        self._labels = self.annotation_df[self.label_col].unique()
        self.label_map = {label: i for i, label in enumerate(self._labels)}
        self.inv_label_map = {i: label for label, i in self.label_map.items()}

        # Filter out the samples that are to be used for training/validation
        self.images_df = self.annotation_df.loc[self.annotation_df['is_valid'] == self.is_validation]
        self.images_df = self.images_df[[self.path_col, self.label_col]]

    def decode_label(self, i):
        """ Decodes the given integer into the label """
        assert i < len(self.label_map), f"Label {i} >= Total number of labels ({len(self.label_map)})"
        return self.inv_label_map[i]

    def encode_label(self, label):
        """ Encodes the givne label to an integer """
        assert label in self.label_map.keys(), f"{label} is not a valid label"
        return self.label_map[label]

    def __len__(self):
        return self.images_df.shape[0]

    def __getitem__(self, idx):
        """
        Returns the image (loaded as tensor, normalized) and its label, located at index 'idx'
        Note that we need to do a transformation, i.e. rescale the dimensions to (224 x 224)
        """
        image_path = str(self.data_dir / self.images_df[self.path_col][idx])
        image_label_name = self.images_df[self.label_col][idx]

        image_tensor = read_image(image_path)                # Loads the image as a uint8 tensor of shape (c, h, w)
        image_tensor = self.transform(image_tensor) / 255.0  # Rescale and normalize the pixel values between 0 and 1
        image_label = self.encode_label(image_label_name)    # Encode the label to an integer

        # It is observed that some images have channels as 1 (grayscale), instead of 3 (rgb)
        # Although this is rare, this can be taken care of by filling the remaining channels with 0
        # Crude method, but does not impact the performance much as such images are rare (hoping so)
        if image_tensor.shape[0] != 3:
            zero_tensor = torch.zeros(3, image_tensor.shape[1], image_tensor.shape[2])
            zero_tensor[0, :] = image_tensor[0, :]
            image_tensor = zero_tensor

        return image_tensor, image_label
