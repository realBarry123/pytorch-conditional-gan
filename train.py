import torch
import numpy

from model import Generator, Discriminator, Classifier, weights_init
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import format_data
import random
# from fetch import fetch_data
from torchvision import datasets, transforms

from tqdm import tqdm
from data import plot_image, one_hot

import matplotlib.pyplot as plt

learning_rate = 0.0002
beta1 = 0.5  # math value, default 0.9
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor()
])

# Download and load the training data
train_set = datasets.MNIST('Datasets/mnist', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Download and load the test data
test_set = datasets.MNIST('Datasets/mnist', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

fixed_noise = torch.randn(128, 100, device="cpu")

# Create our heroes
netG = Generator(0).to("cpu")
netD = Discriminator(0).to("cpu")
netC = Classifier(0).to("cpu")

netC.load_state_dict(torch.load("Models/netC.pkl"))
netC.eval()

try:
    netG.load_state_dict(torch.load("Models/netG.pkl"))  # load netG weights
    netD.load_state_dict(torch.load("Models/netD.pkl"))  # load netD weights

except FileNotFoundError:
    netG.apply(weights_init)
    netD.apply(weights_init)

# define optimizers
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# always remember to instantiate your loss
loss = torch.nn.BCELoss()
CE_loss = torch.nn.CrossEntropyLoss()

test_labels = []

for i in range(128):
    test_labels.append(i%10)

test_labels = torch.tensor(test_labels)

fake = netG(fixed_noise, test_labels).detach().numpy()

#for i in range(10):
    #plot_image(fake[i])

plt.close()

print("-=!Goblin Mode Activated!=-")

for epoch in range(20):

    # for each batch in the dataloader
    for i, data in enumerate(train_loader, start=0):

        real = data[0].to("cpu")
        real_labels = data[1].to("cpu")
        z = torch.randn(128, 100, device="cpu")

        # ========== TRAIN DISCRIMINATOR ==========

        netD.zero_grad()

        # Real image loss

        output = netD(real, real_labels).view(-1)  # Forward pass real batch through D
        label = torch.full((128,), 1.0, dtype=torch.float, device="cpu")

        errD_real = loss(output, label)  # Calculate loss on real batch

        errD_real.backward()

        errD_real_average = output.mean().item()

        # Fake image loss

        label.fill_(0.0)

        fake = netG(z, real_labels)
        output = netD(fake.detach(), real_labels).view(-1)

        errD_fake = loss(output, label)  # Calculate loss on fake batch

        errD_fake.backward()

        errD_fake_average = output.mean().item()

        # Add everything

        errD = errD_real + errD_fake
        print("\nnetD loss:", errD.item())
        optimizerD.step()

        # ========== TRAIN GENERATOR ==========

        netG.zero_grad()

        label.fill_(1.0)

        output = netD(fake, real_labels).view(-1)
        fake = torch.unsqueeze(fake, 1)
        classification = netC(fake)

        errG = loss(output, label) + CE_loss(classification, one_hot(label.long())) * 0.5
        errG.backward()

        errG_average = output.mean().item()

        print("netG loss:", errG_average)

        optimizerG.step()

        torch.save(netD.state_dict(), "Models/netD.pkl")
        torch.save(netG.state_dict(), "Models/netG.pkl")

print("-=.Goblin Mode Deactivated.=-")
