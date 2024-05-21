"""
Barry Yu
2024.04.13
Sign Language Conditional GAN
"""

import torch
import numpy

train_data = numpy.genfromtxt("sign_mnist/train.csv", delimiter=',')
test_data = numpy.genfromtxt("sign_mnist/test.csv", delimiter=',')


def one_hot(_label):
    new_label = numpy.zeros(26)
    new_label[int(_label)] = 1
    return new_label


def format_data(_data):

    # create new arrays
    labels = numpy.empty((len(_data)-1))
    images = numpy.empty((len(_data)-1, 28, 28))

    # separate labels from images
    for i in range(len(_data)-1):
        labels[i] = _data[i+1][0]
        images[i] = numpy.reshape(numpy.delete(_data[i+1], 0), (28, 28))

    # squishification
    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(len(images[i][j])):
               images[i][j][k] /= 255

    # convert to torch tensors with correct types
    labels = torch.tensor(labels).int()
    images = torch.tensor(images).long()

    return labels, images


train_labels, train_images = format_data(train_data)
test_labels, test_images = format_data(test_data)


# print(train_images.shape)  # (27455, 28, 28)
# print(train_labels.shape)  # (27455, 26)
# print(test_images.shape)  # (7172, 28, 28)
# print(test_labels.shape)  # (7172, 26)
