import os
import random

import numpy as np
import skimage.io as imio
from tqdm import tqdm
from PIL import Image

RAW_PHOTO_DIR = "data/photos_raw"
SET_PHOTO_DIR = "data/photos_set"
RAW_PAINTING_DIR = "data/paintings_raw"
SET_PAINTING_DIR = "data/paintings_set"
DATA_RESOLUTION = (1024, 1024)


def build():
    """Runs the generator to convert raw images to dataset images."""
    _build_set("photo", RAW_PHOTO_DIR, SET_PHOTO_DIR, DATA_RESOLUTION)
    _build_set("painting", RAW_PAINTING_DIR, SET_PAINTING_DIR, DATA_RESOLUTION)


def load(shuffle=True):
    """
    Loads the dataset images as numpy matrices.
    :return: (np((N_photos, w, h, 3)), np((N_paintings, w, hH, 3))) where w, h = DATA_RESOLUTION
    """
    photo_set = _load_set("photo", SET_PHOTO_DIR, shuffle)
    painting_set = _load_set("painting", SET_PAINTING_DIR, shuffle)
    return photo_set, painting_set


def _build_set(name, src_dir, dst_dir, target_resolution):
    """
    For every .jpg file in the source directory equal to or larger than the target resolution, downscales the image as
    much as possible without going below the crop resolution, crops the image (center-preserving) to the given
    dimensions, and saves it to the destination directory.

    :param name: Name of the dataset (for printing purposes).
    :param src_dir: Path to source directory containing .jpg files only.
    :param dst_dir: Path to destination directory where downscaled and cropped images will be written to.
    :param target_resolution: (w, h) where w = "Width of the outputted images", h = "Height of the outputted images".
    :return: n: Number of images written to the destination directory.
    """
    w_, h_ = target_resolution
    n = 0
    for img_name in tqdm(os.listdir(src_dir), desc="Building %s set" % name):
        src = "%s/%s" % (src_dir, img_name)
        dst = "%s/%s.jpg" % (dst_dir, str(n).zfill(4))
        img = Image.open(src)
        w, h = img.size
        if w >= w_ and h >= h_:
            f = w / w_ if w < h else h / h_
            w, h = (int(w/f), int(h/f))
            img_downscaled = img.resize((w, h))
            img_cropped = img_downscaled.crop((w / 2 - w_ / 2, h / 2 - h_ / 2,
                                               w / 2 + w_ / 2, h / 2 + h_ / 2))
            img_cropped.save(dst)
            n += 1
    return n


def _load_set(name, src_dir, shuffle=True):
    """
    Loads every .jpg file in the source directory into a numpy array with shape (N, w, h, 3) where
    N = #.jpg files in src_dir
    w, h = DATA_RESOLUTION
    Shuffles the dataset if shuffle = True.

    :param name: Name of the dataset (for printing purposes).
    :param src_dir: Path to source directory containing .jpg files only.
    :param shuffle: if True: shuffles the dataset.
    :return: np((N, w, h, 3)), explained above.
    """
    img_list = list()
    for set_img in tqdm(os.listdir(src_dir), desc="Loading %s set" % name):
        img_list.append(imio.imread("%s/%s" % (src_dir, set_img)))
    if shuffle:
        random.shuffle(img_list)
    return np.array(img_list)
