import pandas as pd
import numpy as np

from . import generator as gen

PHOTO_SRC_DIR = "data/raw_photos"
PHOTO_DST_DIR = "data/training_set"
PAINTING_SRC_DIR = "data/raw_paintings"
PAINTING_DST_DIR = "data/testing_set"
INPUT_W = INPUT_H = 1024


def build():
    gen.filter_and_crop(PHOTO_SRC_DIR, PHOTO_DST_DIR, INPUT_W, INPUT_H)
    gen.filter_and_crop(PAINTING_SRC_DIR, PAINTING_DST_DIR, INPUT_W, INPUT_H)


def load():
    pass
