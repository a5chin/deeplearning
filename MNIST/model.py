from torch import nn
from collections import OrderedDict

classes = 10


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 32),
            nn.LeakyReLU(),

            nn.Linear(32, classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(
            OrderedDict([
                # 32x32
                ('conv1', nn.Conv2d(1, 32, 3, 1, 1)),
                ('batch1', nn.BatchNorm2d(32)),
                ('relu1', nn.LeakyReLU()),
                ('pool1', nn.MaxPool2d(3, padding=1)),

                # 11x11
                ('conv2', nn.Conv2d(32, 64, 3, 1, 1)),
                ('batch2', nn.BatchNorm2d(64)),
                ('relu2', nn.LeakyReLU()),
                ('pool2', nn.MaxPool2d(3, padding=1))

                # 4x4
            ])
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 32),
            nn.LeakyReLU(),

            nn.Linear(32, classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 64 * 4 * 4)
        return self.fc(x)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Bilinear') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
        if m.bias is not None:
            nn.init.zeros_(tensor=m.bias)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
        if m.bias is not None:
            nn.init.zeros_(tensor=m.bias)

    elif classname.find('BatchNorm') != -1 or classname.find('GroupNorm') != -1 or classname.find('LayerNorm') != -1:
        nn.init.uniform_(a=0, b=1, tensor=m.weight)
        nn.init.zeros_(tensor=m.bias)

    elif classname.find('Cell') != -1:
        nn.init.xavier_uniform_(gain=1, tensor=m.weiht_hh)
        nn.init.xavier_uniform_(gain=1, tensor=m.weiht_ih)
        nn.init.ones_(tensor=m.bias_hh)
        nn.init.ones_(tensor=m.bias_ih)

    elif classname.find('RNN') != -1 or classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for w in m.all_weights:
            nn.init.xavier_uniform_(gain=1, tensor=w[2].data)
            nn.init.xavier_uniform_(gain=1, tensor=w[3].data)
            nn.init.ones_(tensor=w[0].data)
            nn.init.ones_(tensor=w[1].data)

    elif classname.find('Embedding') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
