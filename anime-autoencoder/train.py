import os
from argparse import ArgumentParser
from dataset import AnimeFaceDataset
from autoencoder import VAE, Encoder, Decoder, ConvEncoder, ConvDecoder
import torch
from torch import tensor
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from torchsummary import summary
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--epochs', dest='epochs', default=200, type=int)
parser.add_argument('--conv', dest='conv', action='store_true')
parser.add_argument('--batch_size', dest='batch_size', default=128, type=int)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Utilizing device {}.".format(device))


def train():

    data = AnimeFaceDataset("data")
    dataloader = DataLoader(data,
                            batch_size=args.batch_size,
                            shuffle=True)
    # Plot a few images
    # plot_images(data[:100])

    if args.conv:
        encoder = ConvEncoder()
        decoder = ConvDecoder()
    else:
        encoder = Encoder()
        decoder = Decoder()
    autoencoder = VAE(encoder, decoder)

    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)

    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    bce_loss = BCELoss()

    # Print model summary
    summary(autoencoder, input_size=(3, 64, 64))

    for e in range(1, args.epochs + 1):
        for i_batch, sample in enumerate(dataloader):
            x, mu, logvar = autoencoder(sample.to(device))

            # Reconstruction Loss
            rec_loss = bce_loss(x, sample.to(device))

            # Kullback-Leibler Divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total Loss
            loss = rec_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % 1000 == 0:
                print("Ep. {0:>3} with {1:>5} batches; {2:5.2f} loss".format(e, i_batch, loss))

        with torch.no_grad():
            rand = torch.rand([100, 64]).to(device)
            save_images(autoencoder.decoder(rand), e)

        # Save models
        os.makedirs("models", exist_ok=True)
        torch.save(autoencoder.state_dict(), os.path.join("models", "autoencoder.pt"))
        torch.save(encoder.state_dict(), os.path.join("models", "encoder.pt"))
        torch.save(decoder.state_dict(), os.path.join("models", "decoder.pt"))
        print("Saved trained models.")


def plot_images(images: list):
    plt.figure()
    grid = utils.make_grid(images, nrow=10)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()


def save_images(images: tensor, episode: int):
    dir = "test_images/"
    os.makedirs(dir, exist_ok=True)
    plt.figure()
    grid = utils.make_grid(images, nrow=10)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.savefig(dir + str(episode) + "_episode.png")
    plt.close()


if __name__ == '__main__':
    train()
