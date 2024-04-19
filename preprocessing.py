"""
Barry Yu
April 13, 2024
Sign Language Conditional GAN
"""

import csv
import pandas
import torchvision
import numpy

train_data = numpy.genfromtxt("sign_mnist/train.csv", delimiter=',')
test_data = numpy.genfromtxt("sign_mnist/test.csv", delimiter=',')


def format_data(_data):
    """
    :param _data: data loaded from the .csv
    :return: (labels, images)
    """
    labels = numpy.empty(len(_data)-1)
    images = numpy.empty((len(_data)-1, 28, 28))

    for i in range(len(_data)-1):
        labels[i] = _data[i+1][0]
        images[i] = numpy.reshape(numpy.delete(_data[i+1], 0), (28, 28))

    return labels, images


train_labels, train_images = format_data(train_data)
test_labels, test_images = format_data(test_data)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
