import os
import pandas as pd
import numpy as np
import torch
from keras.preprocessing.image import load_img
from torch.utils.data import Dataset

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Custom Dataset class based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# And using various transforms provided by:
#   https://pytorch.org/docs/stable/torchvision/transforms.html

LABELS = ["Pneumonia", "Edema", "Cardiomegaly", "Consolidation", "Pneumothorax", "Atelectasis", "No Finding"]


class KaggleDataset(Dataset):
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
        im = load_img(os.path.join(self.root_dir, image_path), target_size=(600, 600), interpolation='nearest')

        label_list = self.df.iloc[idx, 1].split("|")
        encoded_labels = [0] * len(LABELS)

        for i in range(len(LABELS)):
            if LABELS[i] in label_list:
                encoded_labels[i] = 1

        if self.label_transform is not None:
            label = self.label_transform(encoded_labels)

        if self.image_transform is not None:
            im = self.image_transform(im)

        return im, torch.from_numpy(np.array(encoded_labels))
