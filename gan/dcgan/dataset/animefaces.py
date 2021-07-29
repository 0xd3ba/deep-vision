# animefaces.py -- Dataset class for Anime Faces dataset

# NOTE: I'm using an extremely small dataset (only 64 images) for testing purposes
#       Dataset Link: https://www.kaggle.com/soumikrakshit/anime-faces

import pathlib
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.transforms import CenterCrop
from torchvision.io import read_image


class AnimeFaces(Dataset):
    """
    Dataset class for Anime-Faces dataset

    NOTE: Each image is an RGB image with dimensions (64, 64)
    """

    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        self.images = list(self.data_dir.glob('*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        """
        Returns the i'th element in the dataset
        """
        img = read_image(str(self.images[i]))
        img = img / 255.0                       # Normalize the pixel intensities to 0 and 1

        return img