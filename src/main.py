import os

import model as model_
import dataset.manager as dataset_

os.chdir("..")

REBUILD_DATASET = False

if REBUILD_DATASET:
    dataset_.build()

data = dataset_.load()

generator = model_.Generator()
discriminator = model_.Discriminator()
