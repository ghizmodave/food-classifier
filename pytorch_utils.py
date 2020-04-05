#==================================================================================#
# Author       : Davide Mariani                                                    #
# Script Name  : pytorch_utils.py                                                  #
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
    This function performs the validation steps done at each epoch.
    Attributes:
    model : the pytorch model to validate
    valid_loader : a torch.utils.DataLoader containing validation data
    criterion : a pytorch loss metric
    device : cpu or cuda depending on the machine specs
    fivecrop : specify if the crop function (which crops the given PIL Image into four corners and the central crop)
               is used with "mean", "max" or not used at all.

    """
    valid_loss = 0.0

    model.eval()

    with torch.no_grad():
        for data, target in valid_loader:

            data, target = data.to(device), target.to(device) # move to GPU

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
    This function performs the full training process over a specified number of epochs.
    It returns a trained model and saves the following files:
    - a states dictionary of the best trained model (evaluated on validation results)
    - a .tar file containing states dictionary, model info and classes dictionary for resuming training
    - a .tar file containing states dictionary, model info and classes dictionary of the best trained model (evaluated on validation results)

    Attributes:
    n_epochs : the number of epochs that the training algorithm should perform
    loaders : a torch.utils.DataLoader containing training and validation data
    model : the pytorch model to train
    optimizer :  a pytorch optimizer
    criterion : a pytorch loss metric
    device : cpu or cuda depending on the machine specs
    path_model : the path in which the outputs will be saved
    fivecrop : specify if the crop function (which crops the given PIL Image into four corners and the central crop)
               is used with "mean", "max" or not used at all.
    lr_scheduler : input for pytorch functions to adjust learning rate (like ReduceLROnPlateau)
    valid_loss_min : the best validation loss registered so far (np.Inf by default)
    start_epoch : number of the first epoch (useful when resuming training - 1 by default)
    train_loss : tracker of the training loss (list)
    valid_loss : tracker of the validation loss (list)
    """

    time_start = time.time() #time tracker for the whole process
    best_epoch = start_epoch #tracker of the best epoch for printed messages (defaults to start_epoch if a better one is not found)

    for epoch in range(start_epoch, start_epoch + n_epochs):

        time_start_epoch = time.time() #time tracker for the single epoch

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
            print("Best validation results so far (previous one was {}) - saving best model in {}".format(round(sorted(valid_loss)[1],3), path_model.replace(".pt", "") + "_best.pt.tar"))
            torch.save(model_status, path_model + ".tar") #saving resume training model data
            torch.save(model_status, path_model.replace(".pt", "") + "_best.pt.tar") #saving best model data
        else:
            if train_loss_epoch <= min(train_loss):
                torch.save(model_status, path_model + ".tar") #saving resume training model data
            else:
                print("Checkpoint was not created at last epoch since both validation of {} and training loss of {} have not been improved.".format(round(valid_loss_min,3), round(min(train_loss),3)))


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
    """
    This function load a model checkpoint for resuming training.
    Attributes:
    model : the pytorch model on which resuming the state
    optimizer :  the optimizer on which resuming the state
    scheduler : the scheduler on which resuming the state
    losslogger : the loss on which resuming the state
    filename : the file from which retrieving model training data
    """
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

        if checkpoint['is_best']: #the parameter 'is_best' defines if the resumed training starts from the best model trained so far (based on validation results)
            print("Starting from the best model trained so far (based on validation results - {:.3f})".format(losslogger))
        else:
            print("Not starting from the best model trained so far (based on validation results)")

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, losslogger, train_loss_tracker, valid_loss_tracker


### Functions for testing

def test(loaders, model, criterion, device):
    """
    This function performs the testing process on a trained model.
    Attributes:
    loaders : a torch.utils.DataLoader containing testing data
    model : a trained pytorch model to test
    criterion : a pytorch loss metric
    device : cpu or cuda depending on the machine specs
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

    print('Test Loss: {:.3f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
