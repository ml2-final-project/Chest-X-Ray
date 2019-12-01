import os
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
import tensorflow as tf
from torch.utils.data import Dataset

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Custom Dataset class based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# And using various transforms provided by:
#   https://pytorch.org/docs/stable/torchvision/transforms.html


label_columns = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

class ChexpertDataset(Dataset):
    def __init__(self,
                 csv_file,
                 root_dir,
                 image_transform=None,
                 label_transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_transform = label_transform
        self.image_transform = image_transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]

        # Updated based on: https://github.com/jfhealthcare/Chexpert/blob/47e1bf7f8ed2b95c8985ae43801ffffde92fa205/data/dataset.py#L104
        # Image is grayscale here
        image = cv2.imread(os.path.join(self.root_dir, image_path), 0)
        image = Image.fromarray(image)

        label = self.df.iloc[idx, 1:]
        if self.label_transform is not None:
            label = self.label_transform(label)

        if self.image_transform is not None:
            im = self.image_transform(image)

        return im, torch.from_numpy(np.array(label[label_columns].values.tolist()))



# label Transforms
class ReplaceNaNTransform:
    def __call__(self, sample):
        return sample[label_columns].fillna(0)


class UZerosTransform:
    def __call__(self, sample):
        sample = sample[label_columns].replace(-1, 0)
        return sample


class UOnesTransform:
    def __call__(self, sample):
        sample = sample[label_columns].replace(-1, 1)
        return sample