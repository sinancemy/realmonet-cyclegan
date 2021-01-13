import torch
from torchvision.utils import make_grid
from PIL import Image
import random
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def tensor_to_image(tensor):
    grid = make_grid(tensor, nrow=8, padding=2, pad_value=0, normalize=True, range=None, scale_each=False)
    return Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


def image_to_tensor(image):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(image)


def preprocess_image(image):
    return transforms.Compose([
        transforms.RandomRotation(degrees=(-120, 120))
    ])(image)


def plot_output(output):
    L = len(output)
    plt.figure(figsize=(L * 2, 4))
    plt.tight_layout(0.1)
    for i, imgs in enumerate(output):
        a, b = imgs
        plt.subplot(2, L, i + 1)
        plt.axis("off")
        plt.imshow(a)
        plt.subplot(2, L, L + i + 1)
        plt.axis("off")
        plt.imshow(b)
    plt.show()


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
