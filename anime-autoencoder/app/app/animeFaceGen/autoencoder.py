import torch
from torch import nn, clamp
from torch.distributions import Normal


class ConvVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, logvar = self.encoder(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        x = self.decoder(z)
        return x, mu, logvar


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=32*8*8, out_features=64)
        self.mu = nn.Linear(in_features=64, out_features=32)
        self.std = nn.Linear(in_features=64, out_features=32)

        self.elu = nn.ELU()
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.elu(x)
        x = self.max_pool(x)
        x = x.view(-1, 32*8*8)
        x = self.fc1(x)
        x = self.elu(x)
        mu = self.mu(x)
        std = self.std(x)
        return mu, std


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=32, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32*8*8)
        self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = x.view(-1, 32, 8, 8)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


