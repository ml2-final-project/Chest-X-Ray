from torch.utils.data import DataLoader

from KaggleDataset import KaggleDataset
from testing_common_utils import predict_on_test_data, model_uzeros, model_uones
from training_common_utils import image_preprocessing


def predict(model_name):
    kaggle_data = KaggleDataset(
        csv_file="../Data/sample_labels.csv",
        root_dir="../Data/sample/images",
        image_transform=image_preprocessing
    )

    kaggle_data_loader = DataLoader(kaggle_data,batch_size=64, pin_memory=True, num_workers=16)

    if model_name == "uzeros":
        return predict_on_test_data(kaggle_data_loader, model_uzeros)
    else:
        return predict_on_test_data(kaggle_data_loader, model_uones)
