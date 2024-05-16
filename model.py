import torch
import torch.nn as nn
import torch.nn.parallel


class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.latent = nn.Sequential(
            nn.Linear(in_features=100, out_features=6272),
            nn.ReLU(),
            nn.Unflatten(0, (7, 7, 128))
        )
        self.label = nn.Sequential(
            PrintShape(),
            nn.Embedding(num_embeddings=1, embedding_dim=50),
            PrintShape(),
            nn.Linear(in_features=50, out_features=49),
            nn.Unflatten(dim=1, unflattened_size=(7, 7))
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_channels=129, out_channels=128, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, bias=False),
        )

    def forward(self, latent_input, label_input):
        latent_output = self.latent(latent_input)
        label_output = self.label(label_input)
        print(latent_output.shape)
        print(label_output.shape)
        concated_tensor = torch.cat((latent_output, label_output), dim=2)
        return self.upscale(concated_tensor)
