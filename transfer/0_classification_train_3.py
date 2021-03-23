#### テーマ：モデルを別ファイルで管理しよう（学習編）

import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score ##############

#### 自分で作ったモデルをインポート
from my_utility import my_models


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epoch = 5

img_transform = transforms.Compose([
    transforms.ToTensor()
])

traindataset = torchvision.datasets.MNIST('./data', transform=img_transform, train=True, download=False)  # 一度端末に保存したらdownloadはFalseにしておきましょう
trainloader = DataLoader(traindataset, batch_size=100, shuffle=True)


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
        nn.init.xavier_uniform_(gain=1, tensor=m.weight_hh)
        nn.init.xavier_uniform_(gain=1, tensor=m.weight_ih)
        nn.init.ones_(tensor=m.bias_hh)
        nn.init.ones_(tensor=m.bias_ih)

    elif classname.find('RNN') != -1 or classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for w in m.all_weights:
            nn.init.xavier_uniform_(gain=1, tensor=w[2].data)
            nn.init.xavier_uniform_(gain=1, tensor=w[3].data)
            nn.init.ones_(tensor=w[0].data)
            nn.init.ones_(tensor=w[1].data)

    if classname.find('Embedding') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)


net = my_models.MyCNN(10)
net.apply(init_weights)
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
criterion = torch.nn.NLLLoss()

print(net)

for epoch in range(num_epoch):
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        ##ここで精度計算
        results = outputs.cpu().detach().numpy().argmax(axis=1)
        running_acc += accuracy_score(labels.cpu().numpy(), results) * len(inputs)
        total += len(inputs)

        running_loss += loss.item()

    running_acc /= total
    print('Loss: ', running_loss, 'ACC :', running_acc, epoch)

print('Finished Training')

torch.save(net.state_dict(), './save_models/mnist_classification.pth')
