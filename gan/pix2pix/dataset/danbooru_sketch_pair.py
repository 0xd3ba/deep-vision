# danbooru_sketch_pair.py -- Dataset class for the "Danbooru Sketch-Pair" dataset from Kaggle
# Link:    https://www.kaggle.com/wuhecong/danbooru-sketch-pair-128x

# NOTE: I'm using an extremely small version of it (only 6 images) for testing purposes
#       The actual dataset is big (~9GB)


import pathlib
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image


class DanbooruSketchPair(Dataset):
    """
    Dataset class for the Danbooru Sketch-Pair dataset.
        - Each image is of size (128, 128)
        - The sketch image has 1 channel (Grayscale image)
        - The source image has 3 channels (RGB image)

    NOTE: The sketch image and the source image have the same names, but stay on different
          directories. This makes life easier, as there is no need for an annotation CSV file that
          does the mapping :)
    """
    def __init__(self, root_dir, sketch_dir, src_dir):
        self.dataset_dir = pathlib.Path(root_dir)
        self.sketch_dir = self.dataset_dir / sketch_dir   # Directory containing the sketches
        self.src_dir = self.dataset_dir / src_dir         # Directory contained the ground-truth, i.e. colored sketches

        self.sketch_imgs = list(self.sketch_dir.glob('*.png'))
        self.src_imgs = list(self.src_dir.glob('*.png'))

    def __len__(self):
        return len(self.sketch_imgs)

    def __getitem__(self, idx):
        """
        Returns the i'th image pair
        """
        sketch_loc = str(self.sketch_imgs[idx])
        src_loc = str(self.src_imgs[idx])

        sketch_tensor = read_image(sketch_loc) / 255.0
        src_tensor = read_image(src_loc) / 255.0

        return sketch_tensor, src_tensor

