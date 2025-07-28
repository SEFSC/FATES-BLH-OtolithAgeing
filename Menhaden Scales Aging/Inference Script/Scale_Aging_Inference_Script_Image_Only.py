import argparse
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import torch
import yaml

class FishTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):

        # Get the directory dataset images
        self.image_dir = image_dir

        # Get the transform methods
        self.transforms = transform


        # Image Name
        self.image_name = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]


    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, str(self.image_name[index]))
        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)

        return image, self.image_name[index]
        

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to configuration yaml file")

args = parser.parse_args()

        
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)


data_dir = config["img_path"]

data_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
test_dataset = FishTestDataset( data_dir, data_transforms)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes = 5)

# Load model - TODO
model.load_state_dict(torch.load(config["model_path"]))

model.eval()    
model.to(device)

import torch

output_path = "inference_results.csv"
file = open(config["out_path"], 'w')
file.write("Image Name, Predicted Age\n")

for images, img_path in tqdm(test_loader):
    images = images.to(device)
    outputs = model(images)
    outputs = torch.squeeze(outputs)
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().detach().numpy()
    for i in range(preds.shape[0]):
        age = str(preds[i])
        if(preds[i] ==4):
            age = "4+"
        file.write("%s,%s\n"%(img_path[i],age))
file.close()
