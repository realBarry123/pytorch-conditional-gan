
import matplotlib.pyplot as plt
import torch
# from fetch import fetch_data
from torchvision import datasets, transforms


def plot_image(image):
    plt.imshow(image, interpolation='none')
    plt.show()


def one_hot(nums):
    zeros = torch.zeros(nums.size(0), 10)
    zeros.scatter_(1, nums.unsqueeze(1), 1)
    return nums


transform = transforms.Compose([
    transforms.ToTensor()
])


def get_dataloader(batch_size, train=True):
    train_set = datasets.MNIST('Datasets/mnist', download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)