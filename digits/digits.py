import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import accuracy_score
from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([
    transforms.ToTensor()
])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=True):
        super().__init__()

        self.digits = load_digits()
        self.transform = transform
        self.type = train

        train_data, test_data, train_target, test_target = train_test_split(self.digits.data, self.digits.target, test_size=0.2)
        self.train, self.test = {'data': train_data, 'target': train_target}, {'data': test_data, 'target': test_target}

    def __getitem__(self, item):
        if self.type:
            data = self.transform(Image.fromarray(np.reshape(self.train['data'][item], (8, 8))))
            label = torch.from_numpy(np.array(self.train['target'][item]))
        else:
            data = self.transform(Image.fromarray(np.reshape(self.test['data'][item], (8, 8))))
            label = torch.from_numpy(np.array(self.test['target'][item]))
        return data, label

    def __len__(self):
        return len(self.train['target']) if self.type else len(self.test['target'])


traindataset = Dataset(transform=transforms)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=100, shuffle=True)

testdataset = Dataset(transform=transforms, train=False)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=100, shuffle=False)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Bilinear') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
        if m.bias is not None: nn.init.zeros_(tensor=m.bias)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
        if m.bias is not None: nn.init.zeros_(tensor=m.bias)

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


net = Net(10)
net.apply(init_weights)
net.to(device)


optimizer = optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.NLLLoss()

now = datetime.now()
# writer = SummaryWriter(log_dir='./log/%s' % now.strftime('%Y-%m-%d-%H-%M-%S'))
writer = SummaryWriter(log_dir='./log/train')
writer_test = SummaryWriter(log_dir='./log/test')

total_epoch = 1000

for epoch in range(total_epoch):
    net.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    total_test = 0
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

    running_acc /= total
    running_loss /= total

    writer.add_scalar('loss', running_loss, epoch + 1)
    writer.add_scalar('accuracy', running_acc, epoch + 1)

    print('Loss: %f, Accuracy: %f' % (running_loss, running_acc))

    net.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for data, target in testloader:
            outputs = net(data)
            results = outputs.cpu().detach().numpy().argmax(axis=1)
            test_acc += accuracy_score(target.cpu().numpy(), results) * len(labels)
            test_loss += loss.item() * len(labels)

            total_test += len(labels)

        test_acc /= total_test
        test_loss /= total_test

    writer_test.add_scalar('loss', test_loss, epoch + 1)
    writer_test.add_scalar('accuracy', test_acc, epoch + 1)

