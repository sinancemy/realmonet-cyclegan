import os
import time

import torch
from torch.utils.data import DataLoader

import dataset as _dataset
import model as _model
import operate as _operate
import util as _util

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

# Hyperparameters
learning_rate = 1e-4
batch_size = 2
epochs = 50
decay_epoch = 20
adam_betas = (0.55, 0.999)
params = {"lr": learning_rate, "batch_size": batch_size, "epochs": epochs, "decay_epoch": decay_epoch, "adam_betas": adam_betas}

# Load data
photo_set = _dataset.RealMoNetDataset(_dataset.PHOTO_SET_DIR)
train_photos, test_photos = torch.utils.data.random_split(photo_set, photo_set.get_split(0.90))
train_photos_loader = DataLoader(train_photos, batch_size=int(batch_size/2), shuffle=True)
test_photos_loader = DataLoader(test_photos, batch_size=1, shuffle=False)

painting_set = _dataset.RealMoNetDataset(_dataset.PAINTING_SET_DIR)
train_paintings, test_paintings = torch.utils.data.random_split(painting_set, painting_set.get_split(0.90))
train_paintings_loader = DataLoader(dataset=train_paintings, batch_size=int(batch_size/2), shuffle=True)
test_paintings_loader = DataLoader(dataset=test_paintings, batch_size=1, shuffle=False)

# Create model
model = _model.RealMoNetModel("RealMoNet-%d" % int(time.time()))

# Train model
_operate.train(model, train_photos_loader, train_paintings_loader, device, params)

# or Load model
# model.load("RealMoNet-Pretrained1")

# Convert and plot test images
painting_set.eval()
photo_set.eval()
A_to_B_output = _operate.convert(model, device, test_photos_loader, None)
B_to_A_output = _operate.convert(model, device, None, test_paintings_loader)
_util.plot_output(A_to_B_output, 8)
_util.plot_output(B_to_A_output, 8)
