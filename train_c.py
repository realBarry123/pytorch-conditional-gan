import torch
from train import train_loader, test_loader
from model import Classifier, weights_init

netC = Classifier(0).to("cpu")

try:
    netC.load_state_dict(torch.load("Models/netG.pkl"))  # load netC weights
except FileNotFoundError:
    netC.apply(weights_init)

for epoch in range(5):

    # for each batch in the dataloader
    for i, data in enumerate(train_loader, start=0):
        images = data[0].to("cpu")
        labels = data[1].to("cpu")

        netC.zero_grad()