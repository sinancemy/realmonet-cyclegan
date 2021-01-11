import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self, beta):
        super.__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0, normalize=True):
        super.__init__()

        self.layers = [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding)]
        if normalize:
            self.layers += [nn.InstanceNorm2d(out_features)]
        self.layers += [nn.ReLU()]

        self.layers = nn.Sequential(self.layers)

    def forward(self, x):
        return self.layers(x)


class TransposeConvolutionalLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0, output_padding=1, normalize=True):
        super.__init__()

        self.layers = [nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, stride=stride,
                                          padding=padding, output_padding=output_padding)]
        if normalize:
            self.layers += [nn.InstanceNorm2d(out_features)]
        self.layers += [nn.ReLU()]

        self.layers = nn.Sequential(self.layers)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super.__init__()

        self.residual = [
            nn.ReflectionPad2d(1),
            ConvolutionalLayer(features, features, 3, 1, 0),
            nn.ReflectionPad2d(1),
            ConvolutionalLayer(features, features, 3, 1, 0)
        ]

    def forward(self, x):
        x_initial = x
        for layer in self.model:
            x = layer(x)
        return x_initial + x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = [
            ConvolutionalLayer(3, 64, 4, 2, 1, False),
            ConvolutionalLayer(64, 128, 4, 2, 1),
            ConvolutionalLayer(128, 256, 4, 2, 1),
            ConvolutionalLayer(256, 512, 4, 2, 1),
            ConvolutionalLayer(512, 1024, 4, 2, 1),
            nn.Conv2d(1024, 1, 4, padding=1),
        ]

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = [nn.ReflectionPad2d(3)]

        self.model += [
            ConvolutionalLayer(3, 64, 7, 1, 0),
            ConvolutionalLayer(64, 128, 3, 2, 1),
            ConvolutionalLayer(128, 256, 3, 2, 1)
        ]

        for _ in range(11):
            self.model += [ResidualBlock(256)]

        self.model += [
            TransposeConvolutionalLayer(256, 128, 3, 2, 1, 1),
            TransposeConvolutionalLayer(128, 64, 3, 2, 1, 1)
        ]

        self.model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        ]

        self.layers = nn.Sequential(self.model)

    def forward(self, x):
        return self.model(x)


class RealMoNetModel:
    def __init__(self, name):
        self.name = name
        self.D_A, self.G_A = Discriminator(), Generator()
        self.D_B, self.G_B = Discriminator(), Generator()


def save(model):
    pass


def load(model):
    pass
