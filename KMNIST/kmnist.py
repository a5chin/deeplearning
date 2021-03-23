import torch
from model import *
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import OrderedDict


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([
    transforms.ToTensor()
    # transforms.RandomApply([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(24),
    #     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    # ]),
    # transforms.Resize((32, 32), interpolation=Image.BICUBIC),
])

traindataset = datasets.KMNIST(root='../data', train=True, download=False, transform=transforms)
trainloader = DataLoader(traindataset, batch_size=100, shuffle=True)

testdataset = datasets.KMNIST(root='../data', train=False, download=False, transform=transforms)
testloader = DataLoader(traindataset, batch_size=100, shuffle=False)


net = Net()
net.apply(init_weights)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.NLLLoss()

writer_train = SummaryWriter(log_dir='./log/train')
writer_test = SummaryWriter(log_dir='./log/test')

total_epoch = 50

for epoch in range(total_epoch):
    net.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    with tqdm(enumerate(trainloader, 0), total=len(trainloader)) as pbar:
        pbar.set_description('[Epoch %d/%d]' % (epoch + 1, total_epoch))

        for _, data in pbar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(images)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            results = outputs.cpu().detach().numpy().argmax(axis=1)

            running_acc += accuracy_score(labels.cpu().numpy(), results) * len(labels)
            running_loss += loss.item() * len(labels)

            total += len(labels)

            pbar.set_postfix(OrderedDict(TrainLoss=running_loss / total, TrainACC=running_acc / total))

    running_acc /= total
    running_loss /= total

    writer_train.add_scalar('loss', running_loss, epoch + 1)
    writer_train.add_scalar('accuracy', running_acc, epoch + 1)
    writer_train.add_image('kmnist_images', make_grid(images, nrow=10))

    print('Loss: %f, Accuracy: %f' % (running_loss, running_acc))

    net.eval()
    test_loss = 0.0
    test_acc = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            results = outputs.cpu().detach().numpy().argmax(axis=1)
            test_acc += accuracy_score(labels.cpu().numpy(), results) * len(labels)
            test_loss += loss.item() * len(labels)

            total += len(labels)

        test_acc /= total
        test_loss /= total

    writer_test.add_scalar('loss', test_loss, epoch + 1)
    writer_test.add_scalar('accuracy', test_acc, epoch + 1)
    writer_test.add_image('kmnist_images', make_grid(images, nrow=10))

features = images.view(-1, 28 * 28)
meta_labels = [str(x) for x in labels.numpy().tolist()]

writer_test.add_embedding(features, metadata=meta_labels, label_img=images)
