from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import torch
import json
import os
import numpy as np

class SyntheticDataset(Dataset):

    def __init__(self, dataset_path, opts, metadata = None, transform=None, is_train = True):
        self.dataset_path = dataset_path
        self.opts = opts

        with open(metadata, "r") as f:
            self.metadata = json.load(f)
        self.latents = np.load(os.path.join(dataset_path,'latent.npy'))
        self.transform = transform


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        latent = self.latents[index]
        index = str(index%len(self.metadata))

        canonical_img_path = os.path.join(self.dataset_path,self.metadata[index]['canonical']['filename'])
        canonical_img = Image.open(canonical_img_path)
        canonical_img = canonical_img.convert('RGB')

        random_img_path = os.path.join(self.dataset_path,self.metadata[index]['random_1']['filename'])
        random_img = Image.open(random_img_path)
        random_img = random_img.convert('RGB')

        if self.transform:
            canonical_img = self.transform(canonical_img)
            random_img = self.transform(random_img)

        canonical_camera_param = torch.Tensor(self.metadata[index]['canonical']['camera_params'])
        random_camera_param = torch.Tensor(self.metadata[index]['random_1']['camera_params'])

        return canonical_img, canonical_camera_param, random_img, random_camera_param, latent