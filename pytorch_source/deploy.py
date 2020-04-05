#==================================================================================#
# Author       : Davide Mariani                                                    #
# Script Name  : deploy.py                                                         #
# Description  : scripts for model deployment in AWS Sagemaker                     #
#==================================================================================#
# This file contains functions implemented for deploying a pytorch convolutional   #
# neural network using AWS Sagemaker                                               #
#==================================================================================#

import os
import pandas as pd
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from torchvision import datasets
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# imports the model in model.py by name
from model import ResNetTransfer

from PIL import Image
import io

import pickle


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    model = ResNetTransfer

    #freezing the parameters
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, model_info["n_classes"])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    # Load the output mapper
    output_mapping_path = os.path.join(model_dir, 'output_mapping.pkl')
    with open(output_mapping_path, 'rb') as f:
        mapping_dict = pickle.load(f)


    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model, mapping_dict


def input_fn(request_body, request_content_type):
    print('Deserializing the input data.')
    if request_content_type == 'application/x-image':
        img_arr = np.array(Image.open(io.BytesIO(request_body)))
        img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        
        input_tr = transforms.Compose([
                transforms.Resize(256),  
                transforms.FiveCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.Compose([
                transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])(crop) for crop in crops]))])
        
        data = input_tr(img)
        
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

    
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return json.dumps({'class' : prediction_output['class'], 'prob' : list(prediction_output['prob']),
                      'class_name': prediction_output['class_name'], 'class_names_mapping': prediction_output['class_names_mapping']}), accept 


def predict_fn(input_data, model):
    # load the image and return the predicted food
    mapping_dict = model[1] #retrieving mapping dictionary
    model = model[0] #retrieving model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pred_output = model(input_data.to(device))
    scores = pred_output.mean(0)
    
    output = torch.argmax(scores).to("cpu").item()
    prob = F.softmax(scores,dim=0).to("cpu").data.numpy()
    
    result = {'class': float(output), 'class_name': str(mapping_dict[int(output)]), 'prob':[float(i) for i in prob], 'class_names_mapping':mapping_dict}
    
    return result
