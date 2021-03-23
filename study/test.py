import torch
import pandas as pd
import random
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import csv


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


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


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root=Path('/Users/a5/datasets/siim-isic-melanoma-classification'), train=True, transform=None):
        super().__init__()

        self.transform = transform
        self.images = []

        anno = root
        if train:
            anno /= 'train.csv'
        else:
            anno /= 'test.csv'

        self.df = pd.read_csv(anno)

    def __getitem__(self, index):
        img = self.transform(
            Image.open(str('/Users/a5/datasets/siim-isic-melanoma-classification/jpeg/test/') + self.df['image_name'][
                index] + '.jpg')
        )
        return img

    def __len__(self):
        return len(self.df)


traindataset = MyDataset(train=False, transform=transforms)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=100)


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.convs = nn.Sequential(
            # 512x512  256x256
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, padding=1),

            # 171x171  86x86
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, padding=1),

            # 57x57  29x29
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, padding=1),

            # 19x19  10x10
        )

        self.tell = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 128 * 10 * 10)
        return self.tell(x)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    label = []
    with torch.no_grad():
        with open('./model2/model2.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for data in tqdm(test_loader):
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)

                for i in range(pred.shape[0]):
                    label.append([pred[i].item()])

            writer.writerows(label)


net = Net(2)
net.load_state_dict(torch.load('./model2/model2.pth'))

test(net, device, trainloader)
