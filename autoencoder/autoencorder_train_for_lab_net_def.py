# https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

import os
import time

num_epoch = 5

# ここでは画像の変換をしています．この例ではtorchのテンソルにしたのちに，平均0.5，標準偏差0.5にしています．だいたい[0,1]に収まるとおもいます．
# カラー画像の場合は[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]などと書きます．0.5には意味ないです．
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# 一度端末に保存したらdownloadはFalseにしておきましょう
traindataset = torchvision.datasets.MNIST('./data', transform=img_transform, train=True, download=False)
trainloader = DataLoader(traindataset, batch_size=100, shuffle=True)


# 最終的なネットの出力がnn.Tanh()なので，[-1,1]になっていますが，それを[0,1]に変更しています．clampでクリップしてます（clampは要らない気がするけど）．
# viewで形を整えます．（バッチ x チャネル x 行 x 列）にするのですが，viewを使ってよしなにしてくれています．　https://qiita.com/kenta1984/items/d68b72214ce92beebbe2
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # an affine operation: y = Wx + b
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16))
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# net = Net().cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # としておけば，
net = Net()
net.to(device) # と書くこともできます．複数のGPUを持っている学部や学科のGPUを使う場合はこちらを使いましょう（多分）．

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

print(net)
# exit(0)


start = time.time()

for epoch in range(num_epoch):
    running_loss = 0.0
    for data in trainloader:
        inputs, _ = data
        inputs = inputs.view(inputs.size(0), -1)
        # inputs = Variable(inputs).cuda()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('epoch [', epoch + 1, '/', num_epoch, '], loss =', running_loss)
    # epochの最後のバッチを復元した結果を画像として復元
    if epoch % 1 == 0:
        pic = to_img(outputs.cpu().data)
        if not os.path.exists('./mlp_img/'):
            os.makedirs('./mlp_img/')
        save_image(pic, './mlp_img/image_' + str(epoch) + '.png')


process_time = time.time() - start
print(process_time)

torch.save(net.state_dict(), './autoencoder.pth')


# データが小さいのでGPUが活きません
# 502.85 sec. Core i5  8400  1050Ti 50 epoch
# 499.31 sec. Core i7  7700K 1080Ti 50 epoch  batch=5000 379.64
# 489.34 sec. Core i7 10700K 2080Ti 50 epoch  batch=5000 359.30
