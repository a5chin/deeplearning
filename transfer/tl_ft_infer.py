#### テーマ：転移学習したモデルを使って評価する
####　ただしデータがないので，バリデーションデータで評価
####　https://yutaroogawa.github.io/pytorch_tutorials_jp/


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, transforms

from sklearn.metrics import confusion_matrix

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = torchvision.datasets.ImageFolder(root='./data/hymenoptera_data/val', transform=transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
# 本来テスト時にはshuffleしなくてもよいのですが，あるバッチにクラス１しか含まれていなくて，そのとき全てのデータがTPとかになると混同行列が１行１列の行列になってしまうのを回避するためにshuffleしておきます

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 重み全体を学習した時のモデル
# model_ft = models.resnet18(pretrained=False)  # ネットワークの形だけ欲しいので重みはロードしません
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
# model_ft.load_state_dict(torch.load('./save_models/fine_tune.pth'))
# model_ft = model_ft.to(device)
# model_ft.eval()

# print(model_ft)

# FC層のみ学習した時のモデル
model_conv = models.resnet18(pretrained=False)  # ネットワークの形だけ欲しいので重みはロードしません
for param in model_conv.parameters():
    param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv.load_state_dict(torch.load('./save_models/transfer.pth'))
model_conv = model_conv.to(device)
model_conv.eval()

# print(model_conv)

c_mat=np.zeros([2,2],dtype=int)

with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model_conv(data)
        pred_cpu = output.cpu().detach().numpy().argmax(axis=1)
        c_mat += confusion_matrix(target.cpu().numpy(), pred_cpu)
        # print(pred_cpu)
        # print(target.cpu().numpy())
        # print(confusion_matrix(target.cpu().numpy(), pred_cpu))

print(c_mat)
print(np.sum(c_mat))
