from torch.utils.data import DataLoader

from CheXpertDataset import ChexpertDataset
from testing_common_utils import predict_on_test_data, model_uzeros, model_uones
from training_common_utils import image_preprocessing, label_preprocessing_uzeros, label_preprocessing_uones

# TODO Should we be transforming the unknown and blanks here?  Or do we want to take into account when it is unknown?




def predict(model_name):
        if(model_name == "uzeros"):
                test_data = ChexpertDataset(
                        csv_file='../Data/CheXpert-v1.0-small/valid.csv',
                        root_dir='../Data',
                        image_transform=image_preprocessing,
                        label_transform=label_preprocessing_uzeros
                )
                test_data_loader = DataLoader(test_data, pin_memory = True, num_workers=16)
                return predict_on_test_data(test_data_loader, model_uzeros)
        else:
                test_data = ChexpertDataset(
                        csv_file='../Data/CheXpert-v1.0-small/valid.csv',
                        root_dir='../Data',
                        image_transform=image_preprocessing,
                        label_transform=label_preprocessing_uones
                )
                test_data_loader = DataLoader(test_data)
                return predict_on_test_data(test_data_loader, model_uones)