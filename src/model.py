import torch
import torch.nn as nn
import os

MODELS_DIR = "models"


class Swish(nn.Module):
    def __init__(self, beta):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            Swish(2.4),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x):
        return x + self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(Swish(0.4))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


class Generator(nn.Module):
    def __init__(self, n_residual):
        super(Generator, self).__init__()

        def encoder_block(in_filters, out_filters, kernel_size, stride=1, padding=0, swish_beta=0.0, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            if swish_beta > 0.0:
                layers.append(Swish(swish_beta))
            else:
                layers.append(nn.ReLU(inplace=True))
            return layers

        def decoder_block(in_filters, out_filters, kernel_size, stride=1, padding=0, swish_beta=0.0, normalize=True):
            layers = [nn.Upsample(scale_factor=2),
                      nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            if swish_beta > 0.0:
                layers.append(Swish(swish_beta))
            else:
                layers.append(nn.ReLU(inplace=True))
            return layers

        layers = list()

        # Encoder
        layers += [
            nn.ReflectionPad2d(3),
            *encoder_block(3, 64, 7),
            *encoder_block(64, 128, 3, stride=2, padding=1),
            *encoder_block(128, 256, 3, stride=2, padding=1)
        ]

        # Transformer
        for _ in range(n_residual):
            layers += [ResidualBlock(256)]

        # Decoder
        layers += [
            *decoder_block(256, 128, 3, stride=1, padding=1),
            *decoder_block(128, 64, 3, stride=1, padding=1),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RealMoNetModel:
    def __init__(self, name):
        self.name = name
        self.D_A = Discriminator()
        self.G_AB = Generator(7)
        self.D_B = Discriminator()
        self.G_BA = Generator(7)

    def save(self):
        if not os.path.exists(os.path.join(MODELS_DIR, self.name)):
            os.mkdir(os.path.join(MODELS_DIR, self.name))
        torch.save(self.D_A.state_dict(), os.path.join(MODELS_DIR, self.name, "D_A.rmn"))
        torch.save(self.G_AB.state_dict(), os.path.join(MODELS_DIR, self.name, "G_AB.rmn"))
        torch.save(self.D_B.state_dict(), os.path.join(MODELS_DIR, self.name, "D_B.rmn"))
        torch.save(self.G_BA.state_dict(), os.path.join(MODELS_DIR, self.name, "G_BA.rmn"))

    def load(self, name):
        self.name = name
        self.D_A.load_state_dict(torch.load(os.path.join(MODELS_DIR, self.name, "D_A.rmn")))
        self.G_AB.load_state_dict(torch.load(os.path.join(MODELS_DIR, self.name, "G_AB.rmn")))
        self.D_B.load_state_dict(torch.load(os.path.join(MODELS_DIR, self.name, "D_B.rmn")))
        self.G_BA.load_state_dict(torch.load(os.path.join(MODELS_DIR, self.name, "G_BA.rmn")))


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
