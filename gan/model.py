from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1 * 28 * 28)
        )

    def forward(self, z):
        return self.layer(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layer(x)
