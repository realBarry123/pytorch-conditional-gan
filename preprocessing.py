"""
Barry Yu
2024.04.13
Sign Language Conditional GAN
"""

import torchvision
import numpy

train_data = numpy.genfromtxt("sign_mnist/train.csv", delimiter=',')
test_data = numpy.genfromtxt("sign_mnist/test.csv", delimiter=',')

def one_hot(_label):
    new_label = numpy.zeros(26)
    new_label[int(_label)] = 1
    return new_label

def format_data(_data):
    """
    Turns raw 2d array into nice arrays of labels and images
    :param _data: data loaded from the .csv
    :return: everyone knows what this function returns
    """

    # create new arrays
    labels = numpy.empty((len(_data)-1, 26))
    images = numpy.empty((len(_data)-1, 28, 28))

    # separate labels from images
    for i in range(len(_data)-1):
        labels[i] = one_hot(_data[i+1][0])
        images[i] = numpy.reshape(numpy.delete(_data[i+1], 0), (28, 28))

    # squishification
    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(len(images[i][j])):
               images[i][j][k] /= 255

    return labels, images


train_labels, train_images = format_data(train_data)
test_labels, test_images = format_data(test_data)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

print(test_labels)