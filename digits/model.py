import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.convs = nn.Sequential(
            # 8x8
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, padding=1),

            # 3x3
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, padding=1),
            # 1x1
        )

        self.f = nn.Sequential(
            nn.Linear(128 * 1 * 1, 64),
            nn.LeakyReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LeakyReLU(),

            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.LeakyReLU(),

            nn.Dropout(0.2),

            nn.Linear(16, n),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 128)
        return self.f(x)
