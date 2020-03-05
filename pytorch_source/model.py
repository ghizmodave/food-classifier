import torch
import torchvision.models as models

#importing pretrained ResNet for transfer learning
ResNetTransfer = models.resnet50(pretrained=True) 
