import torch
import numpy

from model import Generator
from preprocessing import format_data

#train_data = numpy.genfromtxt("sign_mnist/train.csv", delimiter=',')
test_data = numpy.genfromtxt("sign_mnist/test.csv", delimiter=',')

#train_labels, train_images = format_data(train_data)
test_labels, test_images = format_data(test_data)

z = torch.randn(100)
netG = Generator(0).to("cpu")
# print(test_labels[0])
# print(torch.tensor(test_labels[0]).long().shape)
print(netG(z, torch.tensor(test_labels[0]).long()))