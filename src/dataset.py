import os

import torch
from torch.utils.data import Dataset
from skimage.io import imread
from PIL import Image
from tqdm import tqdm

PHOTO_RAW_DIR = "data/photos_raw"
PHOTO_SET_DIR = "data/photos_set"
PAINTING_RAW_DIR = "data/paintings_raw"
PAINTING_SET_DIR = "data/paintings_set"
DATA_RESOLUTION = (1024, 1024)


class RealMoNetDataset(Dataset):
    def __init__(self, set_dir):
        self.set_dir = set_dir

    def __len__(self):
        return len(os.listdir(self.set_dir))

    def __getitem__(self, i):
        image = imread(os.path.join(self.set_dir, "%s.jpg" % str(i).zfill(4)))
        return torch.from_numpy(image).float() / 255

    def get_split(self, train_percentage):
        train = int(self.__len__() * train_percentage)
        test = self.__len__() - train
        return [train, test]


def build():
    """Runs the generator to convert raw images to dataset images."""
    _build_set(PHOTO_RAW_DIR, PHOTO_SET_DIR, DATA_RESOLUTION)
    _build_set(PAINTING_RAW_DIR, PAINTING_SET_DIR, DATA_RESOLUTION)


def _build_set(src_dir, dst_dir, target_resolution):
    """
    For every .jpg file in the source directory equal to or larger than the target resolution, downscales the image as
    much as possible without going below the crop resolution, crops the image (center-preserving) to the given
    dimensions, and saves it to the destination directory.

    :param src_dir: Path to source directory containing .jpg files only.
    :param dst_dir: Path to destination directory where downscaled and cropped images will be written to.
    :param target_resolution: (w, h) where w = "Width of the outputted images", h = "Height of the outputted images".
    :return: n: Number of images written to the destination directory.
    """
    w_, h_ = target_resolution
    n = 0
    for img_name in tqdm(os.listdir(src_dir), desc="Building set from %s" % src_dir):
        src = os.path.join(src_dir, img_name)
        dst = os.path.join(dst_dir, "%s.jpg" % str(n).zfill(4))
        img = Image.open(src)
        w, h = img.size
        if w >= w_ and h >= h_:
            f = w / w_ if w < h else h / h_
            w, h = (int(w / f), int(h / f))
            img_downscaled = img.resize((w, h))
            img_cropped = img_downscaled.crop((w / 2 - w_ / 2, h / 2 - h_ / 2,
                                               w / 2 + w_ / 2, h / 2 + h_ / 2))
            img_cropped.save(dst)
            n += 1
    return n
