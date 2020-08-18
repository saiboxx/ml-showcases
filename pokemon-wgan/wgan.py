import os
from collections import OrderedDict
from typing import Callable, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam, RMSprop, Optimizer
from tqdm import tqdm
from skimage import io
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import \
    Compose, ToPILImage, Resize, CenterCrop, ToTensor
from torchvision import utils
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class PokemonDataset(Dataset):
    def __init__(self):
        self.root = 'data/mgan_dataset/mgan_dataset/all'

        self.images = []
        self.transform_load = Compose([ToPILImage(), Resize(64), CenterCrop(64), ToTensor()])
        files = os.listdir(self.root)

        print('Loading Images...')
        for f in tqdm(files):
            try:
                path = os.path.join(self.root, f)
                image = self.transform_load(io.imread(path))
                if image.shape[0] == 3:
                    self.images.append(image)

            except ValueError:
                pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx]


# GENERATOR AND ENCODER STRUCTURE IS TAKEN FROM PYTORCH DCGAN TUTORIAL
class Generator(nn.Module):
    def __init__(self, latent_size: int):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        ngf = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x: tensor):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: tensor):
        return self.main(x)


class WGAN(pl.LightningModule):
    def __init__(self,
                 batch_size: int,
                 latent_size: int,
                 num_workers: int):

        super(WGAN, self).__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_size=latent_size)
        self.discriminator = Discriminator()

        self.generator.apply(self.weights_init)
        self.generator.apply(self.weights_init)

    def weights_init(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, z: tensor):
        return self.generator(z)

    def train_dataloader(self) -> DataLoader:
        dataset = PokemonDataset()
        return DataLoader(dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        generator_opt = RMSprop(self.generator.parameters(), lr=5e-5)
        discriminator_opt = RMSprop(self.discriminator.parameters(), lr=5e-5)
        return [generator_opt, discriminator_opt]

    def sample_z(self, size: Optional[int] = None):
        if size is None:
            z = torch.randn([self.hparams.batch_size, self.hparams.latent_size, 1, 1])
        else:
            z = torch.randn([size, self.hparams.latent_size, 1, 1])

        if self.on_gpu:
            z = z.cuda()
        return z

    def training_step(self, batch, batch_nb, optimizer_idx):
        metric_dict = {}

        if optimizer_idx == 0 and batch_nb % 5 == 0:
            z = self.sample_z()
            x_hat = self.generator(z)

            # Generator Loss
            loss = -torch.mean(self.discriminator(x_hat))
            metric_dict.update({'Gen Loss': loss})
        elif optimizer_idx == 1:
            z = self.sample_z()
            x_hat = self.generator(z)

            # Discriminator Loss
            loss = torch.mean(self.discriminator(x_hat)) - torch.mean(self.discriminator(batch))
            metric_dict.update({'Disc Loss': loss})

        else:
            loss = tensor(0.0, requires_grad=True)

        output = {
            'loss': loss,
            'log': metric_dict
        }
        return output

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            second_order_closure: Optional[Callable] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False):

        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()

        if optimizer_idx == 1:
            optimizer.step()
            optimizer.zero_grad()

            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

    def training_epoch_end(self, training_step_outputs):
        with torch.no_grad():
            z = self.sample_z(25)
            x_hat = self.generator(z)

        plt.figure()
        grid = utils.make_grid(x_hat, nrow=5)
        plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.ioff()
        self.logger.experiment.log({'Generator': plt})
        plt.close()
        return {}


if __name__ == '__main__':
    wgan = WGAN(
        batch_size=32,
        latent_size=100,
        num_workers=4
    )

    wandb_logger = WandbLogger(project='pokemon-wgan')
    wandb_logger.watch(wgan, log='gradients', log_freq=100)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=10,
                         logger=wandb_logger
                         )
    trainer.fit(wgan)
