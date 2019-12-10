from torch.utils.data import DataLoader

from CheXpertDataset import ChexpertDataset
from testing_common_utils import predict_on_test_data, model_uzeros, model_uones
from training_common_utils import image_preprocessing, label_preprocessing_uzeros, label_preprocessing_uones
import numpy as np
from torch.utils.data import Subset


def predict(model_name):
        if(model_name == "uzeros"):
                test_data = ChexpertDataset(
                        csv_file='../Data/CheXpert-v1.0-small/train.csv',
                        root_dir='../Data',
                        image_transform=image_preprocessing,
                        label_transform=label_preprocessing_uzeros
                )
                indices = list(range(len(test_data)))
                np.random.shuffle(indices)

                # Downsample to 30% of available training data
                indices = indices[:int(len(test_data) * .3)]

                test_data = Subset(test_data, indices)
                test_data_loader = DataLoader(test_data, batch_size=128, pin_memory=True, num_workers=16)
                return predict_on_test_data(test_data_loader, model_uzeros)
        else:
                test_data = ChexpertDataset(
                        csv_file='../Data/CheXpert-v1.0-small/train.csv',
                        root_dir='../Data',
                        image_transform=image_preprocessing,
                        label_transform=label_preprocessing_uones
                )
                indices = list(range(len(test_data)))
                np.random.shuffle(indices)

                # Downsample to 30% of available training data
                indices = indices[:int(len(test_data) * .3)]

                test_data = Subset(test_data, indices)
                test_data_loader = DataLoader(test_data, batch_size=64, pin_memory=True, num_workers=16)
                return predict_on_test_data(test_data_loader, model_uones)
