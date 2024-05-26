import torch
import numpy

from model import Generator, Discriminator, weights_init
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import format_data

from tqdm import tqdm

import matplotlib.pyplot as plt


def plotImage(image):
    plt.imshow(image, interpolation='none')
    plt.show()


learning_rate = 0.0002
beta1 = 0.5  # math value, default 0.9
batch_size = 128

train_data = numpy.genfromtxt("sign_mnist/train.csv", delimiter=',')
test_data = numpy.genfromtxt("sign_mnist/test.csv", delimiter=',')

train_labels, train_images = format_data(train_data)
test_labels, test_images = format_data(test_data)

train_data = TensorDataset(train_images, train_labels)
test_data = TensorDataset(test_images, test_labels)

fixed_noise = torch.randn(128, 100, device="cpu")

# Create our heroes
netG = Generator(0).to("cpu")
netD = Discriminator(0).to("cpu")

try:
    netG.load_state_dict(torch.load("Models/netG.pkl"))  # load netG weights
    netD.load_state_dict(torch.load("Models/netD.pkl"))  # load netD weights
except FileNotFoundError:
    netG.apply(weights_init)
    netD.apply(weights_init)
except:
    print(":(")

# print(netD(train_images[0], train_labels[0]))
# plotImage(netG(z, test_labels[0]).detach().numpy())

# create dataloader
dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, drop_last=True)

# define optimizers
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# always remember to instantiate your loss
loss = torch.nn.BCELoss()


plotImage(netG(fixed_noise, test_labels[:128]).detach().numpy()[0])

print("-=!Goblin Mode Activated!=-")

for epoch in range(1):

    # for each batch in the dataloader
    for i, data in enumerate(dataloader, start=0):
        print(i)

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

        print(errD_fake_average)
        print(errD_real_average)

        # Add everything

        errD = errD_real + errD_fake
        optimizerD.step()

        # ========== TRAIN GENERATOR ==========

        netG.zero_grad()

        label.fill_(1.0)

        output = netD(fake, real_labels).view(-1)

        errG = loss(output, label)
        errG.backward()

        errG_average = output.mean().item()

        optimizerG.step()

    torch.save(netD.state_dict(), "Models/netD.pkl")
    torch.save(netG.state_dict(), "Models/netG.pkl")

print("-=.Goblin Mode Deactivated.=-")
