#==================================================================================#
# Author       : Davide Mariani                                                    #
# Script Name  : pytorch_utils.py                                                 #
# Description  : utils for pytorch model training                                  #
#==================================================================================#
# This file contains functions implemented for training a convolutional neural     #
# network in pytorch, with the option of resuming the training and progressing it  #
# across multiple sessions.                                                        #
#==================================================================================#

import time
import datetime
import numpy as np
import os

import torch


### Functions for training

def train_epoch(model,train_loader,optimizer,criterion,device):
    """
    This function performs the training steps done at each epoch.
    Attributes:
    model : the pytorch model to train
    train_loader : a torch.utils.DataLoader containing training data
    optimizer :  a pytorch optimizer
    criterion : a pytorch loss metric
    device : cpu or cuda depending on the machine specs
    """
    train_loss = 0.0

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device) # move to GPU

        optimizer.zero_grad() # set gradients to 0

        output = model(data) # get output

        loss = criterion(output, target) # calculate loss
        train_loss += loss.item() * data.size(0)

        loss.backward() # calculate gradients

        optimizer.step() # take step

    train_loss = train_loss / len(train_loader.dataset)
    return model, train_loss

def valid_epoch(model, valid_loader, criterion, device, fivecrop):
    """
    This function performs the validation steps done at each epoch
    """
    valid_loss = 0.0

    model.eval()

    with torch.no_grad():
        for data, target in valid_loader:

            data, target = data.to(device), target.to(device) # move to GPU

            # if we do test time augmentation with 5crop we'll have an extra dimension in our tensor
            if fivecrop == "mean":
                bs, ncrops, c, h, w = data.size()
                output = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
                output = output.view(bs, ncrops, -1).mean(1)
            elif fivecrop == "max":
                bs, ncrops, c, h, w = data.size()
                output = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
                output = output.view(bs, ncrops, -1).max(1)[0]
            else:
                output = model(data)

            ## update the average validation loss
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

    valid_loss = valid_loss / len(valid_loader.dataset)
    return valid_loss


def train(n_epochs, loaders, model, optimizer, criterion, device, path_model,
    fivecrop = None, lr_scheduler = None, valid_loss_min = np.Inf,
    start_epoch=1, train_loss = [], valid_loss = []):
    """
    model training
    """

    time_start = time.time()
    best_epoch = start_epoch

    for epoch in range(start_epoch, start_epoch + n_epochs):

        time_start_epoch = time.time()

        # train current epoch
        model, train_loss_epoch = train_epoch(model,loaders["train"],optimizer,criterion,device)
        train_loss.append(train_loss_epoch)

        # validate current epoch
        valid_loss_epoch = valid_epoch(model,loaders["val"],criterion,device,fivecrop)

        # learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(valid_loss_epoch)
        valid_loss.append(valid_loss_epoch)

        is_best = False

        if valid_loss_epoch <= valid_loss_min: # save if validation loss is the lowest so far
            torch.save(model.state_dict(), path_model)
            valid_loss_min = valid_loss_epoch
            best_epoch = epoch
            is_best = True

        #epoch time tracker
        minutes_past_epoch = (time.time() - time_start_epoch)//60
        seconds_spare_epoch = round((time.time() - time_start_epoch)%60)
        currentDT = datetime.datetime.now()
        exact_time =  str(currentDT.hour) + ":" + str(currentDT.minute) + ":" + str(currentDT.second)

        # print epoch stats
        print('Epoch {} done in {} minutes and {} seconds at {}. \tTraining Loss: {:.3f} \tValidation Loss: {:.3f}'.format(
            epoch,
            minutes_past_epoch,
            seconds_spare_epoch,
            exact_time,
            train_loss_epoch,
            valid_loss_epoch
            ))

        #save the best model status for resuming training
        model_status = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_loss_min': min(train_loss),
                'valid_loss_min': valid_loss_min,
                'is_best': is_best}

        if lr_scheduler is not None:
            model_status['scheduler'] = lr_scheduler.state_dict()

        if is_best:
            print("Best validation results so far - saving best model in {}".format(path_model.replace(".pt", "") + "_best.pt.tar"))
            torch.save(model_status, path_model + ".tar") #saving resume training model data
            torch.save(model_status, path_model.replace(".pt", "") + "_best.pt.tar") #saving best model data
        else:
            if train_loss_epoch <= min(train_loss):
                torch.save(model_status, path_model + ".tar") #saving resume training model data
            else:
                print("Last epoch was not saved in {} since both validation and training loss didn't improve.")


    #total time tracker
    minutes_past = (time.time() - time_start)//60
    hours_past = minutes_past//60
    minutes_spare = minutes_past%60
    seconds_spare = round((time.time() - time_start)%60)

    # print final statistics
    print(f"{n_epochs} epochs trained in {hours_past} hours, {minutes_spare} minutes and {seconds_spare} seconds. ")

    print("Best model obtained at epoch {} with minimum validation loss : {:.3f}".format(best_epoch, valid_loss_min))

    # Load best config
    model.load_state_dict(torch.load(path_model))

    return model

### Functions for resuming the training

def load_checkpoint(model, optimizer, scheduler, losslogger, filename='models/model_res_101cat.pt.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)

        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler.load_state_dict(checkpoint['scheduler'])

        losslogger = checkpoint['valid_loss_min']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))

        train_loss_tracker = checkpoint['train_loss']
        valid_loss_tracker = checkpoint['valid_loss']

        if checkpoint['is_best']:
            print("Starting from the best model trained so far (based on validation results)")
        else:
            print("Not starting from the best model trained so far (based on validation results)")

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, losslogger, train_loss_tracker, valid_loss_tracker


### Functions for testing

def test(loaders, model, criterion, device):
    """
    test function
    """

    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(loaders['test']):

            data, target = data.to(device), target.to(device) # move to GPU

            bs, ncrops, c, h, w = data.size()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
            output = output.view(bs, ncrops, -1).mean(1)

            loss = criterion(output, target) # calculate the loss

            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss)) # update average test loss

            pred = output.data.max(1, keepdim=True)[1] # convert output probabilities to predicted class

            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
