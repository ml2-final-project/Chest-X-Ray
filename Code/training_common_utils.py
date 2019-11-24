import torch
import torch.nn as nn
import time


# function for performing forward propagation
def simple_forward_propagation(model, optimizer, criterion, local_image, local_label):
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
        device,
        num_epochs
):

    # loss per epoch aggregation variables
    loss_training_agg = []
    loss_validation_agg = []
    for epoch in range(num_epochs):
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

        loss_training_agg.append(loss_training_calc)
        loss_validation_agg.append(loss_validation_calc)

        print("\tEpoch Train Loss {:.5f} - Validation Loss {:.5f}".format(
            loss_training_calc, loss_validation_calc))
        print("\tEpoch Compute Time: {}".format(
            time.time() - epoch_start))
    return model, loss_training_agg, loss_validation_agg
