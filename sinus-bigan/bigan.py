from collections import OrderedDict

import matplotlib.pyplot as plt
from itertools import chain
import torch
import torch.nn as nn
from torch import tensor
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SinusDataset(Dataset):
    def __init__(self, sin_range: int, seq_length: int):
        self.sin_range = sin_range
        self.seq_length = seq_length
        self.x = torch.linspace(0, self.sin_range, self.seq_length)

    def __len__(self):
        return 16000

    def __getitem__(self, _):
        a = (3 - -3) * torch.rand(1) + -3
        b = (3 - -3) * torch.rand(1) + -3
        c = (5 - -5) * torch.rand(1) + -5
        d = (1 - -1) * torch.rand(1) + -1

        return a * torch.sin(b * self.x + c) + d


class Generator(nn.Module):
    def __init__(self, latent_size: int):
        super(Generator, self).__init__()
        self.latent_size = latent_size

        self.generate = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.latent_size,
                               out_channels=256,
                               kernel_size=4,
                               stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(in_channels=64,
                               out_channels=32,
                               kernel_size=4,
                               stride=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(in_channels=32,
                               out_channels=32,
                               kernel_size=5,
                               stride=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(in_channels=32,
                               out_channels=32,
                               kernel_size=1,
                               stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(in_channels=32,
                               out_channels=1,
                               kernel_size=1,
                               stride=1)
        )

    def forward(self, input: tensor):
        output = self.generate(input)
        return output


class Encoder(nn.Module):
    def __init__(self, latent_size: int):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        self.encode = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=4,
                      stride=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=4,
                      stride=1),
            nn.LeakyReLU(inplace=True)
        )

        self.fc1 = nn.Linear(in_features=256 * 16,
                             out_features=latent_size)

    def forward(self, x: tensor):
        batch_size = x.shape[0]
        x = self.encode(x)
        x = self.fc1(x.view(batch_size, -1))
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_size: int):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size

        self.inference_x = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=1),
            nn.LeakyReLU(inplace=True)
        )

        self.inference_joint = nn.Sequential(
            nn.Linear(64 * 16 + latent_size, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: tensor, z: tensor):
        batch_size = x.shape[0]

        output_x = self.inference_x(x).view(batch_size, -1)
        output = self.inference_joint(torch.cat([output_x, z.squeeze(-1)], dim=1))
        return output.squeeze()


class BiGAN(pl.LightningModule):
    def __init__(self,
                 batch_size: int,
                 sin_range: int,
                 seq_length: int,
                 latent_size: int,
                 num_workers: int):

        super(BiGAN, self).__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_size=latent_size)
        self.encoder = Encoder(latent_size=latent_size)
        self.discriminator = Discriminator(latent_size=latent_size)

        self.distribution = Normal(0, 1)

        self.real_label = torch.ones(batch_size).cuda()
        self.fake_label = torch.zeros(batch_size).cuda()
        self.bce_loss = nn.BCELoss()


    def sample_normal(self):
        z = self.distribution.sample([self.hparams.batch_size, self.hparams.latent_size]).unsqueeze_(-1)
        if self.on_gpu:
            z = z.cuda()
        return z

    def forward(self, z: tensor):
        return self.generator(z)

    def train_dataloader(self) -> DataLoader:
        dataset = SinusDataset(self.hparams.sin_range, self.hparams.seq_length)
        return DataLoader(dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        generator_opt = Adam(chain(self.generator.parameters(), self.encoder.parameters()))
        discriminator_opt = Adam(self.discriminator.parameters())
        return [generator_opt, discriminator_opt]

    def training_step(self, batch, batch_nb, optimizer_idx):
        x = batch.view(self.hparams.batch_size, 1, -1)
        z_hat = self.encoder(x)

        z = self.sample_normal()
        x_hat = self.generator(z)

        disc_enc = self.discriminator(x, z_hat)
        disc_gen = self.discriminator(x_hat, z)

        if optimizer_idx == 0:
            # Generator Loss
            loss = self.bce_loss(disc_gen, self.real_label) + self.bce_loss(disc_enc, self.fake_label)
        else:
            # Discriminator Loss
            loss = self.bce_loss(disc_enc, self.real_label) + self.bce_loss(disc_gen, self.fake_label)

        output = OrderedDict({
            'loss': loss,
            #'progress_bar': {'Loss': 0},
            'log': {'Loss': loss}
        })
        return output


if __name__ == '__main__':
    # latent_size = 32
    # dist = Normal(0, 1)
    # gen = Generator(latent_size=latent_size)
    # enc = Encoder(latent_size=latent_size)

    # sample = dist.sample([5, latent_size]).unsqueeze_(-1)

    # generated = gen(sample)
    # print(generated.shape)

    # encoded = enc(generated)
    # print(encoded.shape)

    # dis = Discriminator(latent_size=latent_size)
    # trueness = dis(generated, sample)
    # print(trueness)
    bigan = BiGAN(
        batch_size=64,
        sin_range=25,
        seq_length=95,
        latent_size=32,
        num_workers=4
    )

    trainer = pl.Trainer(gpus=1,
                         max_epochs=100)
    trainer.fit(bigan)
