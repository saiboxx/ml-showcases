import torch
from torch import nn


class VAE(nn.Module):
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


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=32*8*8, out_features=64)
        self.mu = nn.Linear(in_features=64, out_features=64)
        self.logvar = nn.Linear(in_features=64, out_features=64)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
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
        mu = self.sigmoid(mu)

        logvar = self.logvar(x)
        logvar = self.sigmoid(logvar)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=64)
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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=64*64*3, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.mu = nn.Linear(in_features=128, out_features=64)
        self.logvar = nn.Linear(in_features=128, out_features=64)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = x.view(-1, 64*64*3)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        x = self.elu(x)
        x = self.fc4(x)
        x = self.elu(x)

        mu = self.mu(x)
        mu = self.sigmoid(mu)

        logvar = self.logvar(x)
        logvar = self.sigmoid(logvar)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=2048)
        self.fc5 = nn.Linear(in_features=2048, out_features=64*64*3)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        x = self.elu(x)
        x = self.fc4(x)
        x = self.elu(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        x = x.view(-1, 3, 64, 64)
        return x
