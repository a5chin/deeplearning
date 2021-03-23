import torch.nn
import torch.optim as optim
from torch.autograd import Variable as Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict

import matplotlib.pyplot as plt

from model import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([
    transforms.ToTensor()
])

traindataset = datasets.MNIST(root='../data', train=True, download=False, transform=transforms)
trainloader = DataLoader(traindataset, batch_size=100, shuffle=True)

G = Generator(dim=100)
G.apply(init_weights)
G.to(device)
optimizer_G = optim.Adam(G.parameters(), lr=1e-3)

D = Discriminator()
D.apply(init_weights)
D.to(device)
optimizer_D = optim.Adam(D.parameters(), lr=1e-3)

criterion = nn.NLLLoss()

total_epoch = 10

for epoch in range(total_epoch):
    with tqdm(enumerate(trainloader, 0), total=len(trainloader)) as pbar:
        pbar.set_description('[Epoch %d/%d]' % (epoch + 1, total_epoch))

        for _, data in pbar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            b_size = labels.shape[0]

            optimizer_D.zero_grad()

            x = torch.randn(b_size, 10*10).to(device)
            fake_images = G(x)

            label_r = torch.ones(b_size, dtype=torch.long)
            label_f = torch.zeros(b_size, dtype=torch.long)

            outputs_r = D(images.view(-1, 1*28*28))
            outputs_f = D(fake_images.detach().view(-1, 1*28*28))

            loss_r = criterion(outputs_r, label_r)  # 順番次第でerror
            loss_f = criterion(outputs_f, label_f)

            loss_dis = loss_r * 10 + loss_f
            loss_dis.backward()

            optimizer_D.step()

            optimizer_G.zero_grad()

            x = torch.rand(b_size, 10*10).to(device)
            fake_images = G(x)
            outputs_gen = D(fake_images)
            print(fake_images[0])
            loss_gen = criterion(outputs_gen, label_r)
            loss_gen.backward()
            optimizer_G.step()

            z = torch.randn(100, 100).to(device)
            test_img = G(z)
            test_img_array = (test_img * 256.).clamp(min=0., max=255.).data.cpu().numpy().reshape(-1, 28, 28)

            pbar.set_postfix(OrderedDict(Loss=loss_dis.item()))

        fig = plt.figure(figsize=(10, 10))
        for i, im in enumerate(test_img_array):
            ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(im, 'gray')
        plt.savefig('./image/' + str(epoch) + '.png')

