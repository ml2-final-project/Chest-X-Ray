import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Custom Dataset class based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# And using various transforms provided by:
#   https://pytorch.org/docs/stable/torchvision/transforms.html


class ChexpertDataset(Dataset):
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
        im = Image.open(os.path.join(self.root_dir, image_path))
        label = self.df.iloc[idx, 1:]

        if self.label_transform is not None:
            label = self.label_transform(label)

        if self.image_transform is not None:
            im = self.image_transform(im)

        return im, label

