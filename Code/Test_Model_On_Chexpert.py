from torch.utils.data import DataLoader

from Code.CheXpertDataset import ChexpertDataset
from Code.testing_common_utils import predict_on_test_data, model_uzeros
from Code.training_common_utils import image_preprocessing, label_preprocessing_uzeros

# TODO Should we be transforming the unknown and blanks here?  Or do we want to take into account when it is unknown?
test_data = ChexpertDataset(
        csv_file='../Data/CheXpert-v1.0-small/valid.csv',
        root_dir='../Data',
        image_transform=image_preprocessing,
        label_transform=label_preprocessing_uzeros
)

test_data_loader = DataLoader(test_data)

predict_on_test_data(test_data_loader, model_uzeros)
