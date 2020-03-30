import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from PIL import Image

def process_image(image):
    """
    This function takes as argument an image uploaded through an html form,
    and return it ready for model submission and prediction.
    """

    img = Image.open(image)

    input_tr = transforms.Compose([
                transforms.Resize(256),
                transforms.FiveCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.Compose([
                transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])(crop) for crop in crops]))])

    data = input_tr(img)

    print("Image processed for prediction...")

    return data
