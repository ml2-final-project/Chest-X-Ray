from torch.utils.data import DataLoader

from Code.KaggleDataset import KaggleDataset
from Code.testing_common_utils import predict_on_test_data
from Code.training_common_utils import image_preprocessing

kaggle_data = KaggleDataset(
    csv_file="../Data/sample/sample_labels.csv",
    root_dir="../Data/sample/images",
    image_transform=image_preprocessing
)

kaggle_data_loader = DataLoader(kaggle_data)

predict_on_test_data(kaggle_data_loader)