import json
import os
from tqdm import tqdm

with open('dataset.json', 'r') as file:
    labels = json.load(file)['labels']

source = '/workspace/ffhq_512_realign_mirrored'

cnt = 0
new_labels = []
for l in tqdm(labels):
    path = os.path.join(source, l[0])
    if os.path.isfile(path):
        new_labels.append(l)
a = {'labels':new_labels}

with open(os.path.join('ffhq_dataset.json'), 'w') as f:
    json.dump(a, f)
