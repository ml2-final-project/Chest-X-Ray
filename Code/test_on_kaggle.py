from KaggleDataset import KaggleDataset
from training_common_utils import image_preprocessing

data = KaggleDataset(
    csv_file="../Data/sample/sample_labels.csv",
    root_dir="../Data/sample/images",
    image_transform=image_preprocessing
)

# TODO For testing, will be removed once we're running model on data
for i in range(10):
    print(data[i])
