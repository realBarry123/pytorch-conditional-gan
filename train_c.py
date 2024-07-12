import torch
from model import Classifier, weights_init
from torchvision import datasets, transforms

batch_size = 64

learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor()
])

netC = Classifier(0).to("cpu")

try:
    netC.load_state_dict(torch.load("Models/netG.pkl"))  # load netC weights
except:
    netC.apply(weights_init)

# Download and load the training data
train_set = datasets.MNIST('Datasets/mnist', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Download and load the test data
test_set = datasets.MNIST('Datasets/mnist', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(netC.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(5):

    # for each batch in the dataloader
    for i, (data, target) in enumerate(train_loader, start=0):
        images = data[0].to("cpu")
        labels = data[1].to("cpu")

        netC.zero_grad()
        output = netC(images)

        loss = loss_function(output, target)  # loss function
        loss.backward()
        optimizer.step()  # optimizer adjusts the network weights