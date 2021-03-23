#### 学習中の精度が見れるようにしよう

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from sklearn.metrics import confusion_matrix
import numpy as np

from my_utility import my_models
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epoch = 5

img_transform = transforms.Compose([
    transforms.ToTensor()
])

testdataset = torchvision.datasets.MNIST('../data', transform=img_transform, train=False, download=False)  # 一度端末に保存したらdownloadはFalseにしておきましょう
testloader = DataLoader(testdataset, batch_size=100, shuffle=True)

net = my_models.MyCNN(10)
net.load_state_dict(torch.load('./save_models/mnist_classification.pth'))
net.to(device)
net.eval()

test_loss = 0
correct = 0
c_mat=np.zeros([10, 10], dtype=int)
with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        pred_cpu = output.cpu().detach().numpy().argmax(axis=1)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        c_mat += confusion_matrix(target.cpu().numpy(), pred_cpu)

test_loss /= len(testloader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))

print(c_mat)

sns.heatmap(c_mat, annot=True, cmap='Blues')

plt.show()


