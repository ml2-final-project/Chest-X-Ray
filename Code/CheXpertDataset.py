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
                 df_transform=None,
                 image_transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.df_transform = df_transform
        self.image_transforms = image_transform
        # dataframe transforms happen at initial load
        if self.df_transform is not None:
            self.df = df_transform(self.df)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        im = Image.open(os.path.join(self.root_dir, image_path))
        return im, self.df.iloc[idx, 1:]

