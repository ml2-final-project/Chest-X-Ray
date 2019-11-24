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
# TODO: Construct custom DataSet classes
# TODO: Should probably load metadata sepparately to avoid
#   additional memory overhead??
data_validation = ()
data_training = ()

# Note: this assumes data_validation and data_training are the same size
indices = list(range(len(data_validation)))
np.random.shuffle(indices)

# Downsample to 30% of available training data
# TODO: downsample after shuffle?
# indices = indices[:int(len(data_validation)*.3)]

# Note: after these two operations, data_validation and data_training
#   will no longer be of equal lengths.
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


# function for performing forward propagation
def simple_forward_propagation(model, optimizer, criterion, local_image, local_label):
    optimizer.zero_grad()
    logits = model(local_image)
    return criterion(logits, local_label)

# function assumes images and labels are pytorch data objects
def minibatch_training_step(
        model,
        images,
        labels,
        forward_propagation,
        optimizer,
        criterion,
        device):
    # move images and labels to device
    local_images, local_labels = images.to(device, dtype=torch.float), \
                                 labels.to(device, dtype=torch.float)

    # forward
    loss = forward_propagation(
        model,
        optimizer,
        criterion,
        local_images,
        local_labels
    )
    print("\t\tLoss: {}".format(loss))

    # backward
    loss.backward()

    # update
    optimizer.step()
    return loss.item()


def evaluation(model, images, labels, criterion):
    local_images, local_labels = images.to(device, dtype=torch.float), \
                                 labels.to(device, dtype=torch.float)
    y_validation_pred = model(local_images)
    return criterion(y_validation_pred, local_labels).item()


# loss per epoch aggregation variables
loss_training_agg = []
loss_validation_agg = []
for epoch in range(N_EPOCHS):
    # Reduce training rate every 10 epochs
    # TODO: Should we?
    # TODO: Similar for momentum parameter?
    # TODO: Could make of torch optim schedulers for this
    # if epoch % 10 == 0:
    #     LR = LR/2
    #     optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENT)

    loss_training = []
    epoch_start = time.time()
    print("Epoch {}:".format(epoch))
    print("\tStarting Training")

    model.train()
    # assuming loader will return image, label tuple
    for i, (images, labels) in enumerate(training_loader):
        print("\tMiniBatch {}:".format(i))
        minibatch_start = time.time()

        minibatch_loss = minibatch_training_step(
            model,
            images,
            labels,
            simple_forward_propagation,
            optimizer,
            criterion,
            device
        )

        loss_training.append(minibatch_loss)
        print("\t\tLoss: {}".format(minibatch_loss))
        print("\t\tCompute Time: {}".format(time.time() - minibatch_start))

    print('\tEvaluating Model...')
    model.eval()
    with torch.no_grad():
        loss_validation = []
        for (images, labels) in validation_loader:
            loss = evaluation(
                model,
                images,
                labels,
                criterion
            )
            loss_validation.append(loss)

    loss_training_calc = sum(loss_training) / len(loss_training)
    loss_validation_calc = sum(loss_validation) / len(loss_validation)

    loss_training_agg.append(loss_training_calc)
    loss_validation_agg.append(loss_validation_calc)

    print("\tEpoch Train Loss {:.5f} - Validation Loss {:.5f}".format(
        loss_training_calc, loss_validation_calc))
    print("\tEpoch Compute Time: {}".format(
        time.time() - epoch_start))
print("Training Complete: {}s", time.time() - training_start)

# save model first...
torch.save(model.state_dict(), 'model_team8.pt')

# let's plot stuff
import matplotlib.pyplot as plt
plt.plot(loss_training_agg, color='green')
plt.plot(loss_validation_agg, color='red')
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy with Logits Loss")
plt.savefig('loss_v_epochs.png')
