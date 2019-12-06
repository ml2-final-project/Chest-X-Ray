import torch
from torchvision.models import densenet121
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_uzeros = "../Models/model_team8_uzeros.pt"
model_uones = "../Models/model_team8_uzeros.pt"


def predict_on_test_data(test_data_loader, model_file):
    # Load model
    model_state_dict = torch.load(model_file)

    model = densenet121(num_classes=14).to(device)

    model.load_state_dict(model_state_dict)

    outputs = torch.from_numpy(np.array([])).float().to(device)
    target_labels = torch.from_numpy(np.array([])).float().to(device)
    # Classify inputs
    for images, labels in test_data_loader:
        local_images, local_labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
        preds = model(local_images)
        outputs = torch.cat((outputs, preds),0)
        target_labels = torch.cat((target_labels, local_labels),0)

    print(preds)
    return outputs, target_labels
