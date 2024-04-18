"""
Barry Yu
April 13, 2024
Sign Language Conditional GAN
"""

import csv
import pandas
import torchvision

train_data = pandas.read_csv("sign_mnist/train.csv")
test_data = pandas.read_csv("sign_mnist/test.csv")

print(train_data)
