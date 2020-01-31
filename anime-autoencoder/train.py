import os
from dataset import AnimeFaceDataset
from autoencoder import ConvVAE, Encoder, Decoder
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from torchsummary import summary
import matplotlib.pyplot as plt

EPOCHS = 100
BATCH_SIZE = 512


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Utilizing device {}.".format(device))


def train():

    data = AnimeFaceDataset("data")
    dataloader = DataLoader(data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    # Plot a few images
    # plot_images(data[:100])


    encoder = Encoder()
    decoder = Decoder()
    autoencoder = ConvVAE(encoder, decoder)

    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)

    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    bce_loss = BCELoss()

    # Print model summary
    # summary(autoencoder, input_size=(3, 64, 64))

    for e in range(EPOCHS):
        for i_batch, sample in enumerate(dataloader):
            x, mu, std = autoencoder(sample)

            # Reconstruction Loss
            rec_loss = bce_loss(x, sample)

            # Kullback-Leibler Divergence
            kl_loss = 0.5 * torch.sum(torch.exp(std) + mu**2 - 1. - std)

            # Total Loss
            loss = rec_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % 10 == 3:
                print("Ep. {0:>3} with {1:>5} batches; {2:5.2f} loss".format(e, i_batch, loss))

    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(autoencoder, os.path.join("models", "autoencoder.pt"))
    torch.save(encoder, os.path.join("models", "encoder.pt"))
    torch.save(decoder, os.path.join("models", "decoder.pt"))

def plot_images(images: list):
    plt.figure()
    grid = utils.make_grid(images, nrow=10)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    train()
