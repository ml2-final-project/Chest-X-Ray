import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from CheXpertDataset import ChexpertDataset, \
    UZerosTransform, UOnesTransform, ReplaceNaNTransform
from ImageTransforms import PILToNumpy, FixRatioResize, EqualizeHistogram, Gray2RGB, GaussianBlur
from torchvision import transforms
from torch.utils.data import Subset, DataLoader

image_preprocessing = transforms.Compose([
    PILToNumpy(),
    EqualizeHistogram(),
    Gray2RGB(),
    FixRatioResize(512, 128),
    GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((128, 128, 128), (64, 64, 64))
])

image_augmented_preprocessing = transforms.Compose([
    transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                            scale=(0.85, 1.05), fillcolor=128),
    image_preprocessing
])

label_preprocessing_uzeros = transforms.Compose([
    UZerosTransform(),
    ReplaceNaNTransform()
])

label_preprocessing_uones = transforms.Compose([
    UOnesTransform(),
    ReplaceNaNTransform()
])

label_preprocessing_umulticlass = ReplaceNaNTransform()


def build_data_loaders(base_label_transform, dlparams):
    # Construct DataSet class for validation and training
    # Note: thinking to keep them separate to support possibility of data augmentations
    data_validation = ChexpertDataset(
        csv_file='../Data/CheXpert-v1.0-small/train.csv',
        root_dir='../Data',
        image_transform=image_preprocessing,
        label_transform=base_label_transform
    )

    data_training = ChexpertDataset(
        csv_file='../Data/CheXpert-v1.0-small/train.csv',
        root_dir='../Data',
        image_transform=image_augmented_preprocessing,
        label_transform=base_label_transform
    )
    # Note: this assumes data_validation and data_training are the same size
    indices = list(range(len(data_validation)))
    np.random.shuffle(indices)

    # Downsample to 30% of available training data
    indices = indices[:int(len(data_validation) * .3)]

    # Note: after these two operations, data_validation and data_training
    #   will no longer be of equal lengths.
    # 30% of downsampled data will be used for validation, and 70% for training
    data_validation = Subset(data_validation, indices[:int(len(indices) * .3)])
    data_training = Subset(data_training, indices[int(len(indices) * .3):])

    training_loader = DataLoader(data_training, **dlparams)
    validation_loader = DataLoader(data_validation, **dlparams)

    return training_loader, validation_loader


# function for performing forward propagation
def simple_forward_propagation(model, optimizer, criterion, local_image, local_label):
    #print(local_image.shape)
    #print(local_label.shape)
    optimizer.zero_grad()
    logits = model(local_image)
    return criterion(logits, local_label)


# function assumes images and labels are pytorch data objects
def simple_minibatch_training_step(
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


def evaluation(model, images, labels, criterion, device):
    local_images, local_labels = images.to(device, dtype=torch.float), \
                                 labels.to(device, dtype=torch.float)
    y_validation_pred = model(local_images)
    return criterion(y_validation_pred, local_labels).item()


def training_loop(
        model,
        training_loader,
        validation_loader,
        forward_propagation,
        minibatch_training_step,
        optimizer,
        criterion,
        scheduler,
        device,
        num_epochs,
        model_name
):
    # Lets initialize loss to some high value
    min_train_loss = min_val_loss = 100

    # loss per epoch aggregation variables
    loss_training_agg = []
    loss_validation_agg = []
    for epoch in range(num_epochs):
        loss_training = []
        epoch_start = time.time()
        print("Epoch {}:".format(epoch))
        print("\tStarting Training")

        model.train()
        # assuming loader will return image, label tuple
        print("The type of training_loader is:", type(training_loader))
        for i, (images, labels) in enumerate(training_loader):
            print("\tMiniBatch {}:".format(i))
            minibatch_start = time.time()
            #print("labels:", labels)

            minibatch_loss = minibatch_training_step(
                model,
                images,
                labels,
                forward_propagation,
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
                    criterion,
                    device
                )
                loss_validation.append(loss)

        loss_training_calc = sum(loss_training) / len(loss_training)
        loss_validation_calc = sum(loss_validation) / len(loss_validation)

        if scheduler is not None:
            scheduler.step(loss_validation_calc)

        loss_training_agg.append(loss_training_calc)
        loss_validation_agg.append(loss_validation_calc)

        print("\tEpoch Train Loss {:.5f} - Validation Loss {:.5f}".format(
            loss_training_calc, loss_validation_calc))
        print("\tEpoch Compute Time: {}".format(
            time.time() - epoch_start))
        if loss_validation_calc < min_val_loss:
            print("*******Loss imrpoved*******")
            print("Saving the new best")
            torch.save(model.state_dict(), model_name)
            min_train_loss = min(min_train_loss, loss_training_calc)
            min_val_loss = min(min_val_loss, loss_validation_calc)

    return model, loss_training_agg, loss_validation_agg
