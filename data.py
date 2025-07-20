
import matplotlib.pyplot as plt
import torch
# from fetch import fetch_data
from torchvision import datasets, transforms


def plot_image(image):
    plt.imshow(image, interpolation='none')
    plt.show()

def plot_multiple(images, h, w):

    fig, axs = plt.subplots(h, w)

    for i, ax in enumerate(axs.flat):  # loop thru positions
        ax.imshow(images[i])

    plt.show()

def one_hot(nums):
    zeros = torch.zeros(nums.size(0), 10)
    zeros.scatter_(1, nums.unsqueeze(1), 1)
    return nums


transform = transforms.Compose([
    transforms.ToTensor()
])


def get_dataloader(batch_size, train=True):
    dataset = datasets.MNIST('Datasets/mnist', download=True, train=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)