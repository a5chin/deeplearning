import torch.nn as nn


class MyCNN(nn.Module):
    """Some Information about MyCNN"""
    def __init__(self, num_classes):
        super().__init__()

        self.convs = nn.Sequential(
            # 28x28  32x32
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, padding=1),

            # 10x10 11x11
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, padding=1),

            # 4x4
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.LeakyReLU(),

            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        h = self.convs(x)  # (batch) x 64 x 4 x 4
        h = h.view(-1, 64 * 4 * 4)  # (batch) x 4
        return self.fc(h)
