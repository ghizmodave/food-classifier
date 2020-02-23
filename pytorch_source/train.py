import argparse
import json
import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.models as models

#importing pretrained ResNet for transfer learning
ResNetTransfer = models.resnet50(pretrained = True)

#freezing the parameters
for param in model_transfer.parameters():
    param.requires_grad = False
