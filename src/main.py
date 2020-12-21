import os
import time

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

import dataset
import model

os.chdir("..")  # Set working directory to main project directory.

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")

time.sleep(1)   # Wait 1 second (for printing purposes).

# Rebuild dataset
REBUILD_DATASET = True
if REBUILD_DATASET:
    dataset.build()

# Hyper-parameters
learning_rate = 2e-3
batch_size = 8
epochs = 1

# Load data
photo_set = dataset.RealMoNetDataset(dataset.PHOTO_SET_DIR)
train_photos, test_photos = torch.utils.data.random_split(photo_set, photo_set.get_split(0.9))
train_photos_loader = DataLoader(train_photos, batch_size=batch_size, shuffle=True)
test_photos_loader = DataLoader(test_photos, batch_size=batch_size, shuffle=True)

painting_set = dataset.RealMoNetDataset(dataset.PAINTING_SET_DIR)
train_paintings, test_paintings = torch.utils.data.random_split(painting_set, painting_set.get_split(0.9))
train_paintings_loader = DataLoader(dataset=train_paintings, batch_size=batch_size, shuffle=True)
test_paintings_loader = DataLoader(dataset=test_paintings, batch_size=batch_size, shuffle=True)

# generator = model.Generator()
# discriminator = model.Discriminator()
