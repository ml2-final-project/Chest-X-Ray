from KaggleDataset import KaggleDataset
from torch.autograd import Variable
from training_common_utils import image_preprocessing
import torch
from torchvision.models import densenet121
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kaggle_data = KaggleDataset(
    csv_file="../Data/sample/sample_labels.csv",
    root_dir="../Data/sample/images",
    image_transform=image_preprocessing
)

kaggle_data_loader = DataLoader(kaggle_data)

# Load model
model_state_dict = torch.load("../Models/model_team8_uzeros_15epoch.pt")

model = densenet121(num_classes=14).load_state_dict(model_state_dict)
model.eval()

# Classify inputs
for i, t in kaggle_data_loader:
    inputs, targets = Variable(i), Variable(t)
    outputs = model(inputs)

