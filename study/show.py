import pandas as pd
import cv2
import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def rgb_hist(rgb_img):
    sns.set()
    sns.set_style(style='ticks')
    fig = plt.figure(figsize=[15, 4])
    ax1 = fig.add_subplot(1, 2, 1)
    sns.set_style(style='whitegrid')
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(rgb_img)

    color = ['r', 'g', 'b']

    for (i, col) in enumerate(color):  # 各チャンネルのhist
        # cv2.calcHist([img], [channel], mask_img, [binsize], ranges)
        hist = cv2.calcHist([rgb_img], [i], None, [256], [0, 256])
        # グラフの形が偏りすぎるので √ をとってみる
        # hist = np.sqrt(hist)
        ax2.plot(hist, color=col)
        ax2.set_xlim([0, 256])

    plt.show()


root = Path('/Users/a5/datasets/siim-isic-melanoma-classification')
T = {'train': root / 'train.csv', 'test': root / 'test.csv'}
images = {'train': root / 'jpeg/train', 'test': root / 'jpeg/test'}

file = pd.read_csv(T['train'])

image = cv2.cvtColor(cv2.imread(str(images['train'] / file['image_name'][0]) + '.jpg'), cv2.COLOR_BGR2RGB)

rgb_hist(image)
