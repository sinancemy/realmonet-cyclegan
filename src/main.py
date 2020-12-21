import os
import time

import dataset as dataset_
import model as model_

os.chdir("..")  # Set working directory to main project directory.
time.sleep(1)   # Wait 1 second (for printing purposes).

# Process the raw images, divide them into training and testing sets, convert to csv.
REBUILD_DATASET = True
if REBUILD_DATASET:
    dataset_.build()

data = dataset_.load(shuffle=True)
print("Dataset loaded!")


# generator = model_.Generator()
# discriminator = model_.Discriminator()
