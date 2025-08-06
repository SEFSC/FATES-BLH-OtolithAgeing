#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Title: Scale_Aging_Inference_Script_Image_Only.py
#
# Description: This script predicts the age of fish scale images using a pre-
#              trained ResNet18 model. Arguments, hyperparameters, and other
#              settings are included in a configs.yml file. Predicted ages are
#              written to a CSV file.
#
# Author: aotian.zheng@noaa.gov
# Release Date: July 2025
# Last Updated: August 2025
#
# Usage: python Scale_Aging_Inference_Script_Image_Only.py -c path/to/configs.yml
# -----------------------------------------------------------------------------

import argparse
import yaml
import cv2 as cv
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

class FishTestDataset(Dataset):
    """Custom Dataset for loading fish scale images for age inference.
    
    Attributes
    ----------
    image_dir : str
        Path to the directory containing images.
    image_name : list
        List of image filenames in the directory.
    transforms : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version.
    
    Methods
    -------
    __len__ : returns the number of images in the dataset.
    __getitem__(index) : returns the image and its filename at the specified index.
    """
    def __init__(self, image_dir, transform=None):
        """
        Parameters
        ----------
        image_dir : str
            Path to the directory containing images.
        transform : callable, optional
            A function/transform that takes in a PIL image and returns a transformed version.
        """

        # Get the directory of the images to age
        self.image_dir = image_dir

        # Get the transform methods
        self.transforms = transform

        # Image Name
        self.image_name = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_name)

    def __getitem__(self, index):
        """Returns the image and its filename at the specified index."""
        # Open the specified image
        img_path = os.path.join(self.image_dir, str(self.image_name[index]))
        image = Image.open(img_path)
        # Transform the image, if transforms are provided
        if self.transforms:
            image = self.transforms(image)

        return image, self.image_name[index]
        
# Parse command line arguments. Currently only requires a path to a configuration yaml file.
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", help="path to configuration yaml file")
args = parser.parse_args()

# Open the configuration file and read in the parameters
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

# Image transformations: resizing, cropping, normalization
data_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
test_dataset = FishTestDataset(image_dir=config["image_path"], transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)

# Load the model using GPU, if available, in evaluation mode.
# Number of classes corresponds to the number of age classes.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes=5)
model.load_state_dict(torch.load(config["model_path"]))
model.eval()    
model.to(device)

# Create output file and write header
file = open(config["out_path"], 'w')
file.write("Image Name, Predicted Age\n")

# Loop through the dataset and make predictions
for images, img_path in tqdm(test_loader):
    images = images.to(device)
    outputs = model(images)
    outputs = torch.squeeze(outputs)
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().detach().numpy()
    for i in range(preds.shape[0]):
        age = str(preds[i])
        # Change the maximum age class to "4+"
        if(preds[i] == 4):
            age = "4+"
        file.write("%s,%s\n" % (img_path[i], age))
file.close()
