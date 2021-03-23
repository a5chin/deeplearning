import torch
import pandas as pd
import random
import numpy as np
from torch import nn
from torch import optim as optim
# from torch.nn import functional as F
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import accuracy_score   # confusion_matrix
from datetime import datetime


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomAffine(degrees=(0, 0), scale=(0.7, 1.3), resample=Image.BICUBIC),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
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
        self.class_labels = np.array(self.df['target'])

    def __getitem__(self, index):
        img = self.transform(
            Image.open(str('/Users/a5/datasets/siim-isic-melanoma-classification/jpeg/train/') + self.df['image_name'][
                index] + '.jpg')
        )

        label = torch.from_numpy(np.array(self.df['target'][index]))

        return img, label

    def __len__(self):
        return len(self.class_labels)


traindataset = MyDataset(train=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=100, shuffle=True)


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.convs = nn.Sequential(
            OrderedDict([
                # 256x256
                ('conv1', nn.Conv2d(3, 32, 3, 1, 1)),
                ('batch1', nn.BatchNorm2d(32)),
                ('relu1', nn.LeakyReLU()),
                ('pool1', nn.MaxPool2d(3, padding=1)),

                # 86x86
                ('conv2', nn.Conv2d(32, 64, 3, 1, 1)),
                ('batch2', nn.BatchNorm2d(64)),
                ('relu2', nn.LeakyReLU()),
                ('pool2', nn.MaxPool2d(3, padding=1)),

                # 29x29
                ('conv3', nn.Conv2d(64, 128, 3, 1, 1)),
                ('batch3', nn.BatchNorm2d(128)),
                ('relu3', nn.LeakyReLU()),
                ('pool3', nn.MaxPool2d(3, padding=1))

                # 10x10
            ])
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512),
            nn.LeakyReLU(),

            nn.Dropout(p=0.25),

            nn.Linear(512, 128),
            nn.LeakyReLU(),

            nn.Dropout(p=0.25),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Dropout(p=0.25),

            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 128 * 10 * 10)
        return self.fc(x)


net = Net(2)
net.apply(init_weights)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.NLLLoss()

now = datetime.now()
writer = SummaryWriter(log_dir='./log/Raw')

total_epoch = 10

for epoch in range(total_epoch):
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    with tqdm(enumerate(trainloader, 0), total=len(trainloader)) as pbar:
        pbar.set_description('[Epoch %d/%d]' % (epoch + 1, total_epoch))
        for i, data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            results = outputs.cpu().detach().numpy().argmax(axis=1)

            running_acc += accuracy_score(labels.cpu().numpy(), results) * len(inputs)
            running_loss += loss.item() * len(inputs)

            total += len(inputs)

            pbar.set_postfix(OrderedDict(Loss=running_loss / total, ACC=running_acc / total))

    # torch.save(net.state_dict(), 'model%d.pth' % (epoch + 1))

    running_acc /= total
    running_loss /= total

    writer.add_scalar('loss', running_loss, epoch)
    writer.add_scalar('accuracy', running_acc, epoch)
    writer.add_graph(net, inputs)
    writer.add_image('melanoma_images', make_grid(inputs, nrow=4))
    writer.add_histogram('conv1.bias', net.convs.conv1.bias, epoch + 1)
    writer.add_histogram('conv1.weight', net.convs.conv1.weight, epoch + 1)
    writer.add_histogram('conv2.bias', net.convs.conv2.bias, epoch + 1)
    writer.add_histogram('conv2.weight', net.convs.conv2.weight, epoch + 1)
    writer.add_histogram('conv3.bias', net.convs.conv3.bias, epoch + 1)
    writer.add_histogram('conv3.weight', net.convs.conv3.weight, epoch + 1)

    print('Loss: %f, Accuracy: %f' % (running_loss, running_acc))

features = inputs.view(-1, 3 * 256 * 256)
meta_labels = [str(x) for x in labels.numpy().tolist()]

writer.add_embedding(features, metadata=meta_labels, label_img=inputs)


# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss = F.nll_loss(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('¥nTest set: Average Loss: {:.4f}, Accuracy{}/{} ({:.0f}%)¥n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
