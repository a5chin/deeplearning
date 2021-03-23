import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1 * 28 * 28),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, dim=1*28*28):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
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
