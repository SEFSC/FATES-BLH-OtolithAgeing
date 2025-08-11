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
import pandas as pd

class FishTestDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):

        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        
        # Get the directory dataset images
        self.image_dir = image_dir

        # Get the transform methods
        self.transforms = transform


        # Image Name
        self.image_name = np.asarray(self.data_info.iloc[:, 0])
        
        # Otolith length
        self.length = np.asarray(self.data_info.iloc[:, 1])

        # Otolith weight
        self.wt = np.asarray(self.data_info.iloc[:, 2])

        # Month
        self.month = np.asarray(self.data_info.iloc[:, 3])
        
        # Fish Age
        self.age = np.asarray(self.data_info.iloc[:, 4])


    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, str(self.image_name[index]))
        image = Image.open(img_path)
        
        wt_l_m = torch.tensor([(self.wt[index] - 163)/(82), (self.length[index] - 211)/ (35.5), (self.month[index]-7.4)/(1.9)]).type(torch.FloatTensor)

        
        if(self.age[index] < 5):
          label_age = self.age[index]
        else:
          label_age = 4
            
        if self.transforms:
            image = self.transforms(image)

        return (image,wt_l_m) , self.image_name[index], label_age
        

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to configuration yaml file")

args = parser.parse_args()

        
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)


data_dir = config["train_img_path"]

data_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
train_dataset = FishTestDataset( data_dir, config["train_csv"], data_transforms)
val_dataset = FishTestDataset( data_dir, config["valid_csv"], data_transforms)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, drop_last=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes = 5).to(device)

# load pretrained model
loaded_state_dict = torch.hub.load_state_dict_from_url("https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth")
current_model_dict = model.state_dict()
new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}

import torch.nn as nn
num_epochs = config["epochs"]
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config["scheduler"], gamma=config["gamma"])
criterion = nn.CrossEntropyLoss()

if(not os.path.exists(config["model_out_path"])):
   os.mkdir(config["model_out_path"])

import copy

best_acc = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_res = []

    running_loss = 0.0
    running_corrects = 0
    running_corr = [0.0, 0.0, 0.0, 0.0, 0.0]
    running_total = [0.0, 0.0, 0.0, 0.0, 0.0]
    for images, imagename, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

    
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output = model(images)#inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
        # statistics
        _, preds = torch.max(output, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

        for i in range(0, len(preds)):
            if labels.data[i].cpu().detach().numpy() == 3:
                count_3 += 1

            if preds[i] == labels.data[i]:
                running_corr[int(labels.data[i].cpu().detach().numpy())] += 1.0
            running_total[int(labels.data[i].cpu().detach().numpy())] += 1.0
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * running_corrects / len(train_loader.dataset)
    running_res = [100.0 * i / max(1,j) for i, j in zip(running_corr, running_total)]
    print(running_res)
    print("{} Loss: {:.4f} Average Accuracy: {:.4f}".format("train", epoch_loss, epoch_acc))


    # Validation phase
    model.eval()
    running_res = []

    running_loss = 0.0
    running_corrects = 0
    running_corr = [0.0, 0.0, 0.0, 0.0, 0.0]
    running_total = [0.0, 0.0, 0.0, 0.0, 0.0]
    for images, imagename, labels in tqdm(val_loader):
    
        images = images.to(device)
        labels = labels.to(device)
    
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            output = model(images)#inputs)
            
        # statistics
        _, preds = torch.max(output, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

        for i in range(0, len(preds)):
            if labels.data[i].cpu().detach().numpy() == 3:
                count_3 += 1

            if preds[i] == labels.data[i]:
                running_corr[int(labels.data[i].cpu().detach().numpy())] += 1.0
            running_total[int(labels.data[i].cpu().detach().numpy())] += 1.0
    scheduler.step()
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100.0 * running_corrects / len(val_loader.dataset)
    running_res = [100.0 * i / max(1,j) for i, j in zip(running_corr, running_total)]
    print(running_res)
    print("{} Loss: {:.4f} Average Accuracy: {:.4f}".format("validation", epoch_loss, epoch_acc))
    if(epoch_acc > best_acc):
        print("saving best model")
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        res = running_res.copy()
        torch.save(model.state_dict(), config["model_out_path"]+'/best_model.pth')

torch.save(model.state_dict(), config["model_out_path"]+'/final_model.pth')

