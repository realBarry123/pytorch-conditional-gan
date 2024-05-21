import torch
import numpy

from model import Generator, Discriminator
from preprocessing import format_data

import matplotlib.pyplot as plt


def plotImage(image):
    plt.imshow(image, interpolation='none')
    plt.show()


# train_data = numpy.genfromtxt("sign_mnist/train.csv", delimiter=',')
test_data = numpy.genfromtxt("sign_mnist/test.csv", delimiter=',')

# train_labels, train_images = format_data(train_data)
test_labels, test_images = format_data(test_data)

z = torch.randn(100)
netG = Generator(0).to("cpu")
netD = Discriminator(0).to("cpu")

# print(netD(torch.tensor(test_images[0]).long(), torch.tensor(test_labels[0]).int()))
# plotImage(netG(z, torch.tensor(test_labels[0]).int()).detach().numpy())

print(netD(test_images[0], test_labels[0].int()))
plotImage(netG(z, test_labels[0]).detach().numpy())