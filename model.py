import torch
import torch.nn as nn
import torch.nn.parallel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # find convolutional layer
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # randomly initialize weights
    elif classname.find('BatchNorm') != -1:  # find batchnorm
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # randomly initialize weights
        nn.init.constant_(tensor=m.bias.data, val=0)  # set bias to 0


class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class MaxPool2d(nn.Module):
    def __init__(self):
        super(MaxPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.max_pool2d(x, kernel_size=2)

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        return nn.functional.log_softmax(x, dim=1)

class Generator(nn.Module):

    def __init__(self, ngpu):

        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.latent = nn.Sequential(
            nn.Linear(in_features=100, out_features=6272),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(7, 7, 128))
        )

        self.label = nn.Sequential(
            PrintShape(),
            nn.Embedding(num_embeddings=10, embedding_dim=50),
            nn.Linear(in_features=50, out_features=49),
            nn.Unflatten(dim=1, unflattened_size=(7, 7, 1)),
        )

        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_channels=129, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, latent, label):

        latent = self.latent(latent)
        label = self.label(label)

        concated_tensor = torch.cat((latent, label), dim=3)

        concated_tensor = concated_tensor.permute(0, 3, 1, 2)  # [128, 7, 7, 129] -> [128, 129, 7, 7]
        return torch.squeeze(self.upscale(concated_tensor))


class Discriminator(nn.Module):
    def __init__(self, ngpu):

        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.label = nn.Sequential(
            nn.Embedding(num_embeddings=10, embedding_dim=50),
            nn.Linear(in_features=50, out_features=784),
            nn.Unflatten(dim=1, unflattened_size=(28, 28)),
        )

        self.discriminate = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(start_dim=1, end_dim=3),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1152, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, latent, label):

        label = self.label(label).unsqueeze(3)
        latent = latent.squeeze(1)
        latent = latent.unsqueeze(3)

        print(label.shape)
        print(latent.shape)

        concated_tensor = torch.cat((latent, label), dim=3)
        concated_tensor = concated_tensor.permute(0, 3, 1, 2)

        return self.discriminate(concated_tensor)

class Classifier(nn.Module):

    def __init__(self, ngpu):

        super(Classifier, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            MaxPool2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.Dropout2d(),
            MaxPool2d(),
            nn.ReLU(),
            nn.Flatten(-1, 320),
            nn.Linear(in_features=320, out_features=50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=50, out_features=10),
            Softmax()
        )
    def forward(self, image):
        return self.main(image)
