import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm, trange

root = Path('/Users/a5/datasets/siim-isic-melanoma-classification')
trains = root / 'jpeg/train'

if __name__ == '__main__':
    df = pd.read_csv(root / 'train.csv')
    minimum = 480

    for train in tqdm(trains.iterdir(), total=len(df)):
        try:
            img = cv2.imread(str(train))
            height, width = img.shape[: 2]
        except:
            pass
        m = height if height < width else width
        minimum = minimum if minimum < m else m

    print(minimum)
