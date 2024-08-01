import torch
from model import Classifier, weights_init
from torchvision import datasets, transforms
from data import plot_image
from tqdm import tqdm

def get_max(values):
    return max(range(len(values)), key=values.__getitem__)

batch_size = 64

learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor()
])

netC = Classifier(0).to("cpu")

try:
    netC.load_state_dict(torch.load("Models/netC.pkl"))  # load netC weights
    print("loaded netC.pkl")
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

test_batch = next(iter(test_loader))
print(get_max(netC(torch.tensor(test_batch[0].to("cpu")))[0]))


def test():
    """
    Tests the network for 1 epoch
    :param epoch: the number of the current epoch
    """
    # set network to evaluation mode
    netC.eval()
    loss_sum = 0
    correct = 0

    for i, data in enumerate(test_loader, start=1):
        images = data[0].to("cpu")
        labels = data[1].to("cpu")
        output = netC(images)

        # update progress bar with loss and accuracy values
        loss_sum += loss_function(output, labels).item() / labels.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).sum() / labels.size(0)

        print(f"val_loss: {loss_sum / i:.4f}, val_accuracy: {correct / i:.4f}")

for epoch in range(5):
    netC.train()
    # for each batch in the dataloader
    for i, data in enumerate(train_loader, start=0):
        images = data[0].to("cpu")
        labels = data[1].to("cpu")
        netC.zero_grad()
        output = netC(images)

        loss = loss_function(output, data[1])  # loss function
        loss.backward()
        optimizer.step()  # optimizer adjusts the network weights

    torch.save(netC.state_dict(), "Models/netC.pkl")
    test()
