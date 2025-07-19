import torch

from model import Generator, Discriminator, Classifier, weights_init
from torch.utils.data import TensorDataset

# from fetch import fetch_data
from torchvision import datasets, transforms

from tqdm import tqdm
from data import one_hot, get_dataloader

learning_rate = 0.0002
beta1 = 0.3  # math value, default 0.9
batch_size = 128

classifier_beta = 0.01  # this should be small like definitely 0.01 or lower

# Download and load the training data
train_loader = get_dataloader(batch_size, train=True)

fixed_noise = torch.randn(128, 100, device="cpu")

# Create our heroes
netG = Generator(0).to("cpu")
netD = Discriminator(0).to("cpu")
netC = Classifier(0).to("cpu")

netC.load_state_dict(torch.load("Models/netC.pkl"))
for param in netC.parameters():
    param.requires_grad = False

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


print("-=!Goblin Mode Activated!=-")

for epoch in range(1):

    pbar = tqdm(enumerate(train_loader, start=0), total= len(train_loader))

    # for each batch in the dataloader
    for i, data in pbar:

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
        optimizerD.step()

        # ========== TRAIN GENERATOR ==========

        netG.zero_grad()

        label.fill_(1.0)

        output = netD(fake, real_labels).view(-1)
        fake = torch.unsqueeze(fake, 1)

        classification = netC(fake)
        err_classifier = CE_loss(classification, one_hot(label.long()))

        errG = loss(output, label) + err_classifier * classifier_beta
        errG.backward()

        errG_average = output.mean().item()

        pbar.set_description(
            f"================================="
            f"\nnetG loss: {errG_average} "
            f"\n    netC loss: {err_classifier.item()}"
            f"\nnetD loss: {errD_fake_average} + {errD_real_average} "
            f"\nprogress"
        )
        optimizerG.step()

        torch.save(netD.state_dict(), "Models/netD.pkl")
        torch.save(netG.state_dict(), "Models/netG.pkl")

print("-=.Goblin Mode Deactivated.=-")
