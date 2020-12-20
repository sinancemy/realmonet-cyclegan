import os

import model as model_
import dataset.manager as dataset_

# Set working directory to main project directory
os.chdir("..")

# Process the raw images, divide them into training and testing sets, convert to csv.
REBUILD_DATASET = True
if REBUILD_DATASET:
    dataset_.generate()
    dataset_.build()

data = dataset_.load()

generator = model_.Generator()
discriminator = model_.Discriminator()
