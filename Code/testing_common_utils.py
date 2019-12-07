import torch
from torchvision.models import densenet121
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_uzeros = "../Models/model_team8_uzeros_v2.pt"
model_uones = "../Models/model_team8_uones_v3.pt"


def predict_on_test_data(test_data_loader, model_file):
    # Load model
    cpu_device = torch.device("cpu")
    model_state_dict = torch.load(model_file)

    model = densenet121(num_classes=14).to(device)

    model.load_state_dict(model_state_dict)

    model.eval()

    with torch.no_grad():
        outputs = torch.from_numpy(np.array([])).float()
        target_labels = torch.from_numpy(np.array([])).float()
        # Classify inputs
        for i, (images, labels) in enumerate(test_data_loader):
            print("predict on test data, minibatch: " + str(i))
            local_images, local_labels = images.to(device, dtype=torch.float),\
                                         labels.to(device, dtype=torch.float)
            local_preds = model(local_images)

            preds = local_preds.to(cpu_device, dtype=torch.float)

            print(preds.shape, preds.dtype)
            print(labels.shape, labels.dtype)

            outputs = torch.cat((outputs, preds), 0)
            target_labels = torch.cat((target_labels, labels.float()), 0)

    return outputs, target_labels
