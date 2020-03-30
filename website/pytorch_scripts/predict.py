import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

import os
import pickle

#config
model_dir = "pytorch_scripts/prod_model/" #path relative to app.py

#loading model's base trained architecture
ResNetTransfer = models.resnet50(pretrained=True)

# First, load the parameters used to create the model.
model_info = {}
model_info_path = os.path.join(model_dir, 'model_info.pth')
with open(model_info_path, 'rb') as f:
    model_info = torch.load(f)

print("model_info: {}".format(model_info))

# Determine the device and construct the model.
device = torch.device("cpu") #("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}.".format(device))

#transfer-learning steps
model = ResNetTransfer
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, model_info["n_classes"])

# Load the stored model parameters.
model_path = os.path.join(model_dir, 'model.pth')
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f, map_location="cpu"))

# Load the output mapper
output_mapping_path = os.path.join(model_dir, 'output_mapping.pkl')
with open(output_mapping_path, 'rb') as f:
    mapping_dict = pickle.load(f)

# set to eval mode, could use no_grad
model.to(device).eval()

print("Model loaded...")


def predict_image(input_data):
    # load the image and return the predicted food
    device = torch.device("cpu") #('cuda' if torch.cuda.is_available() else 'cpu')

    pred_output = model(input_data.to(device))
    scores = pred_output.mean(0)

    output = torch.argmax(scores).to("cpu").item()
    prob = F.softmax(scores,dim=0).to("cpu").data.numpy()

    result = {'class': float(output), 'class_name': str(mapping_dict[int(output)]), 'prob':[float(i) for i in prob], 'class_names_mapping':mapping_dict}

    return result
