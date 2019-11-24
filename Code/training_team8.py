# Starting from sample code:
#   https://github.com/amir-jafari/Deep-Learning/blob/master/Pytorch_/CNN/1_ImageClassification/example_MNIST.py
# We then decided to attempt using a densenet121 network based on the original paper
#   that was put out with the dataset:
#   https://arxiv.org/abs/1901.07031
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import densenet121
from .CheXpertDataset import ChexpertDataset
from .training_common_utils import training_loop
from .training_common_utils import simple_forward_propagation as forward_propagation
from .training_common_utils import simple_minibatch_training_step as minibatch_training_step
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 5e-3
MOMENT = .9
BATCH_SIZE = 8
N_EPOCHS = 50


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Custom Dataset class based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# And using various transforms provided by:
#   https://pytorch.org/docs/stable/torchvision/transforms.html

# TODO: Data Augmentation?
#   flipping probably wouldn't work well.
#   neither would rotation, unless fairly small rotations?
#   Color adjustments?

preprocessing = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
])

# Construct DataSet class for validation and training
# Note: thinking to keep them separate to support data augmentations
data_validation = ChexpertDataset(
    csv_file='../Data/train.csv',
    root_dir='../Data',
    image_transform=preprocessing
)

data_training = ChexpertDataset(
    csv_file='../Data/train.csv',
    root_dir='../Data',
    image_transform=preprocessing
)

# Note: this assumes data_validation and data_training are the same size
indices = list(range(len(data_validation)))
np.random.shuffle(indices)

# Downsample to 30% of available training data
indices = indices[:int(len(data_validation)*.3)]

# Note: after these two operations, data_validation and data_training
#   will no longer be of equal lengths.
# 30% of downsampled data will be used for validation, and 70% for training
data_validation = Subset(data_validation, indices[:int(len(indices)*.3)])
data_training = Subset(data_training, indices[int(len(indices)*.3):])

# dataloader parameters
params = {'batch_size': BATCH_SIZE,
          'shuffle': True,  # why not.. *shrug*?
          'num_workers': 12}

training_loader = DataLoader(data_training, **params)
validation_loader = DataLoader(data_validation, **params)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = densenet121(num_classes=14).to(device)

# TODO: Hyperparameter training for best LR, momentum settings?
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=.9)

# TODO: Need to update this? possibly not.
criterion = nn.BCEWithLogitsLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Training Starting")
training_start = time.time()

model, training_losses, validation_losses = training_loop(
    model,
    training_loader,
    validation_loader,
    forward_propagation,
    minibatch_training_step,
    optimizer,
    criterion,
    device,
    N_EPOCHS
)

print("Training Complete: {}s", time.time() - training_start)

# save model first...
torch.save(model.state_dict(), 'model_team8.pt')

# let's plot stuff
import matplotlib.pyplot as plt
plt.plot(training_losses, color='green')
plt.plot(validation_losses, color='red')
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy with Logits Loss")
plt.savefig('loss_v_epochs.png')
