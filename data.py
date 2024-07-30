
import matplotlib.pyplot as plt
import torch
def plot_image(image):
    plt.imshow(image, interpolation='none')
    plt.show()

def one_hot(nums):
    zeros = torch.zeros(nums.size(0), 10)
    zeros.scatter_(1, nums.unsqueeze(1), 1)
    return nums
