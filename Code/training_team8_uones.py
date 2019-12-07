# Starting from sample code:
#   https://github.com/amir-jafari/Deep-Learning/blob/master/Pytorch_/CNN/1_ImageClassification/example_MNIST.py
# We then decided to attempt using a densenet121 network based on the original paper
#   that was put out with the dataset:
#   https://arxiv.org/abs/1901.07031
import torch
import torch.nn as nn
import time
from torchvision.models import densenet121
from torch.optim import lr_scheduler
from training_common_utils import training_loop
from training_common_utils import simple_forward_propagation as forward_propagation
from training_common_utils import simple_minibatch_training_step as minibatch_training_step
from training_common_utils import build_data_loaders
from training_common_utils import label_preprocessing_uones as base_label_transform
import numpy as np

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.0001
MOMENT = .9
BATCH_SIZE = 8
N_EPOCHS = 15

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Custom Dataset class based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# And using various transforms provided by:
#   https://pytorch.org/docs/stable/torchvision/transforms.html
data_loader_params = {
    'batch_size': BATCH_SIZE,
    'num_workers': 16,
    'pin_memory': True}

training_loader, validation_loader = build_data_loaders(
    base_label_transform,
    data_loader_params
)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = densenet121(num_classes=14).to(device)

# comment this out to train the initial model.
model.load_state_dict(torch.load('../Models/model_team8_uones_v2.pt'))

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENT)

criterion = nn.BCEWithLogitsLoss()

scheduler = lr_scheduler.StepLR(optimizer, 2, 0.1)

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
    scheduler,
    device,
    N_EPOCHS,
    "model_team8_uones_v3.pt"
)

print("Training Complete: {}s", time.time() - training_start)

# let's plot stuff
import matplotlib.pyplot as plt
plt.plot(training_losses, color='green')
plt.plot(validation_losses, color='red')
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy with Logits Loss")
plt.savefig('loss_v_epochs_uones_v3.png')
