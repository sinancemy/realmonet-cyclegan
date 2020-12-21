import os
import time

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

import dataset as _dataset
import model as _model
import operate as _operate

os.chdir("..")  # Set working directory to main project directory.

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU :)")
else:
    device = torch.device("cpu")
    print("Running on CPU :(")

time.sleep(1)  # For clean printing.

# Rebuild dataset
REBUILD_DATASET = False
if REBUILD_DATASET:
    _dataset.build()

# Hyper-parameters
learning_rate = 2e-3
batch_size = 8
epochs = 1

# Load data
photo_set = _dataset.RealMoNetDataset(_dataset.PHOTO_SET_DIR)
train_photos, test_photos = torch.utils.data.random_split(photo_set, photo_set.get_split(0.95))
train_photos_loader = DataLoader(train_photos, batch_size=batch_size, shuffle=True)
test_photos_loader = DataLoader(test_photos, batch_size=1, shuffle=False)

painting_set = _dataset.RealMoNetDataset(_dataset.PAINTING_SET_DIR)
train_paintings, test_paintings = torch.utils.data.random_split(painting_set, painting_set.get_split(0.95))
train_paintings_loader = DataLoader(dataset=train_paintings, batch_size=batch_size, shuffle=True)
test_paintings_loader = DataLoader(dataset=test_paintings, batch_size=1, shuffle=False)

model = _model.RealMoNetModel("RealMoNet-%d" % int(time.time()))

_operate.train(model, train_photos_loader, train_paintings_loader)

_operate.convert(model, dl_A=test_photos_loader, dl_B=None)
_operate.convert(model, dl_A=None, dl_B=test_paintings_loader)
