import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Model working on GPU")
else:
    device = torch.device("cpu")
    print("Model working on CPU")


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass
