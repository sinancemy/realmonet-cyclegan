import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import dataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass
