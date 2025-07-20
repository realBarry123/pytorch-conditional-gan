from data import *
from model import Generator

fixed_noise = torch.randn(128, 100, device="cpu")

netG = Generator(0).to("cpu")
netG.load_state_dict(torch.load("Models/netG.pkl"))  # load netG weights

test_labels = []

for i in range(128):
    test_labels.append(i%10)

test_labels = torch.tensor(test_labels)

fake = netG(fixed_noise, test_labels).detach().numpy()

plot_multiple(fake, 4, 10)

plt.close()
