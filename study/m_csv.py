import csv

with open('/Users/a5/datasets/melanoma/HAM10000_metadata.csv', 'r') as f:
    reader = csv.reader(f)
    data = [x for x in reader]

with open('/Users/a5/datasets/siim-isic-melanoma-classification/train.csv', 'r') as m:
    reader = csv.reader(m)
    ori = [x for x in reader]
    ori.append(data)

with open('/Users/a5/datasets/siim-isic-melanoma-classification/train.csv', 'w') as m:
    writer = csv.writer(m)
    writer.writerows(ori)
