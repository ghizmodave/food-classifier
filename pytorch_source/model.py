import torch
import torchvision.models as models

#importing pretrained ResNet for transfer learning
ResNetTransfer = models.resnet50().load_state_dict(torch.load("resnet50_base.pt"))
