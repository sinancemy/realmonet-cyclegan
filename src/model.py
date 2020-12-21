import torch
import torch.nn as nn


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


class RealMoNetModel:
    def __init__(self, name):
        self.name = name
        self.D_A, self.G_A = Discriminator(), Generator()
        self.D_B, self.G_B = Discriminator(), Generator()


def save(model):
    pass


def load(model):
    pass
