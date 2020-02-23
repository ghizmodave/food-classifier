import argparse
import json
import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data

# imports the model in model.py by name
from model import ResNetTransfer


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print('CUDA is not available.  Training on CPU ...')
        device = "cpu"
    else:
        print('CUDA is available!  Training on GPU ...')
        device = torch.device("cuda")
        print("Using",torch.cuda.get_device_name(device))

    model = ResNetTransfer

    #freezing the parameters
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, model_info["n_classes"])

    # Initialize the weights of the new layer
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity='relu')

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model
