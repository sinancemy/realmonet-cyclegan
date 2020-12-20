import pandas as pd
from . import generator as gen

RAW_PHOTO_DIR = "data/generator/raw_photos"
SET_PHOTO_DIR = "data/generator/set_photos"
RAW_PAINTING_DIR = "data/generator/raw_paintings"
SET_PAINTING_DIR = "data/generator/set_paintings"
DATA_RES = (1024, 1024)


def generate():
    """Runs the generator to convert raw images to dataset images."""
    gen.process_directory(RAW_PHOTO_DIR, SET_PHOTO_DIR, DATA_RES)
    gen.process_directory(RAW_PAINTING_DIR, SET_PAINTING_DIR, DATA_RES)


def build():
    """Divides the dataset images into testing and training sets creates CSV files for the sets with the images."""
    # TODO: Implement
    pass


def load():
    """
    Loads the CSV files for the testing and training sets.
    :return: dataset. TODO: Elaborate
    """
    # TODO: Implement
    return None
