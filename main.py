import torch
import numpy

from model import Generator, Discriminator
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import format_data

from tqdm import tqdm

import matplotlib.pyplot as plt


def plotImage(image):
    plt.imshow(image, interpolation='none')
    plt.show()


learning_rate = 0.0002
beta1 = 0.5  # math value, default 0.9
batch_size = 128

train_data = numpy.genfromtxt("sign_mnist/train.csv", delimiter=',')
test_data = numpy.genfromtxt("sign_mnist/test.csv", delimiter=',')

train_labels, train_images = format_data(train_data)
test_labels, test_images = format_data(test_data)

train_data = TensorDataset(train_images, train_labels)
test_data = TensorDataset(test_images, test_labels)

z = torch.randn(100)
fixed_noise = torch.randn(64, 100, 1, 1, device="cpu")
netG = Generator(0).to("cpu")
netD = Discriminator(0).to("cpu")

# print(netD(test_images[0], test_labels[0].int()))
# plotImage(netG(z, test_labels[0]).detach().numpy())

# create dataloader
dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True)

# define optimizers
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

print("-=!Goblin Mode Activated!=-")

