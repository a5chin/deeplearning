import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict

from model import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.ToTensor()
])

mnist_data = datasets.MNIST(root='../data', train=True, transform=transforms, download=False)
training_data = DataLoader(mnist_data, batch_size=100, shuffle=True)


latent_dim = 100
G = Generator(latent_dim=latent_dim).to(device)
D = Discriminator().to(device)
opt_g = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_d = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))

writer_g = SummaryWriter(log_dir='./log/loss_g')
writer_d = SummaryWriter(log_dir='./log/loss_d')

n_epoch = 10
n_critic = 2
criterion = nn.BCEWithLogitsLoss()

for epoch in range(1, n_epoch + 1):
    Tensor = torch.FloatTensor
    with tqdm(enumerate(training_data, 0), total=len(training_data)) as pbar:
        pbar.set_description('[Epoch %d/%d]' % (epoch, n_epoch))
        for idx, (real_x, _) in pbar:
            real_x = real_x.to(device)
            batch = real_x.size(0)
            flag_real = Tensor(batch, 1).fill_(1.0)
            flag_fake = Tensor(batch, 1).fill_(0.0)

            for _ in range(n_critic):
                D.zero_grad()
                z = torch.randn(batch, latent_dim).to(device)
                fake_x = G(z)
                out_real = D(real_x.view(batch, -1))
                out_fake = D(fake_x.detach().view(batch, -1))
                loss_real = criterion(out_real, flag_real)
                loss_fake = criterion(out_fake, flag_fake)
                dis_loss = loss_real + loss_fake
                dis_loss.backward()
                opt_d.step()

            G.zero_grad()
            z = torch.randn(batch, latent_dim).to(device)
            fake_x = G(z)
            out_gen = D(fake_x)
            gen_loss = criterion(out_gen, flag_real)
            gen_loss.backward()
            opt_g.step()

            pbar.set_postfix(OrderedDict(Loss=dis_loss.item()))
        # z = torch.randn(100, 100).to(device)
        # test_img = G(z)
        # test_img_array = (test_img * 256.).clamp(min=0., max=255.).data.cpu().numpy().reshape(-1, 28, 28)

    # writer_d.add_scalar('dis_loss', dis_loss.item(), epoch)
    # writer_g.add_scalar('gen_loss', gen_loss.item(), epoch)


            # if idx % 100 == 0:
            #     print('Training epoch: {} [{}/{} ({:.0f}%)] | D loss: {:.6f} | G loss: {:.6f} |' \
            #           .format(epoch, idx * len(real_x), len(training_data.dataset),
            #                   100. * idx / len(training_data), dis_loss.item(), gen_loss.item()))

z = torch.randn(100, 100).to(device)
test_img = G(z)
test_img_array = (test_img * 256.).clamp(min=0., max=255.).data.cpu().numpy().reshape(-1, 28, 28)

fig = plt.figure(figsize=(10, 10))
for i, im in enumerate(test_img_array):
    ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[])
    ax.imshow(im, 'gray')

plt.show()
