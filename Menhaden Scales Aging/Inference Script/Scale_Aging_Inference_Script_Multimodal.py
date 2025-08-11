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
        
        if self.transforms:
            image = self.transforms(image)

        return (image,wt_l_m) , self.image_name[index]
        

# define multimodal resnet architecture
import torch  # core library for tensor computation and deep neural networks
import torchvision  # working with image and video data
import torch.nn as nn  # neural network modules and loss functions
from PIL import Image  # image manipulation
from torch import Tensor  # tensor representation
import torch.optim as optim  # optimization algorithms
import torch.nn.functional as F  # functional operations in neural networks
from torchvision import datasets  # standard datasets
import torchvision.transforms as T  # image transformations
from torchvision import transforms  # image transformations
from torchvision.models import ResNet, resnet18  # pre-defined ResNet models
from torch.utils.data import DataLoader  # loading data in PyTorch
from torchvision.transforms import ToTensor  # converting PIL images to tensors
from torch.utils.data.dataset import Dataset  # creating custom datasets
from torch.hub import load_state_dict_from_url  # model loading
from typing import Any, Callable, List, Optional, Type, Union, Tuple  # type hints for code understanding
# function creating a 3x3 convolutional layer
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
    Function to create a 3x3 convolutional layer with padding.

    Args:
    - in_planes (int): Number of input channels.
    - out_planes (int): Number of output channels.
    - stride (int): Stride value for the convolution (default: 1).
    - groups (int): Number of groups for grouped convolution (default: 1).
    - dilation (int): Dilation rate for the convolution (default: 1).

    Returns:
    - conv_layer (nn.Conv2d): The created 3x3 convolutional layer.
    """
    # Create a 3x3 convolutional layer with the specified parameters
    conv_layer = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

    return conv_layer

# function creating a 1x1 convolutional layer
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    Function to create a 1x1 convolutional layer.

    Args:
    - in_planes (int): Number of input channels.
    - out_planes (int): Number of output channels.
    - stride (int): Stride value for the convolution (default: 1).

    Returns:
    - conv_layer (nn.Conv2d): The created 1x1 convolutional layer.
    """
    # Create a 1x1 convolutional layer with the specified parameters
    conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    return conv_layer

# module to define a BasicBlock residual block for the resnet model
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Basic residual block implementation used in ResNet.

        Args:
        - inplanes (int): Number of input channels.
        - planes (int): Number of output channels.
        - stride (int): Stride value for the convolutional layers (default: 1).
        - downsample (nn.Module, optional): Downsample module (default: None).
        - groups (int): Number of groups for grouped convolution (default: 1).
        - base_width (int): Base width for grouped convolution (default: 64).
        - dilation (int): Dilation rate for dilated convolution (default: 1).
        - norm_layer (Callable[..., nn.Module], optional): Normalization layer (default: nn.BatchNorm2d).
        """
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)  # 3x3 convolutional layer
        self.bn1 = norm_layer(planes)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation function
        self.conv2 = conv3x3(planes, planes)  # 3x3 convolutional layer
        self.bn2 = norm_layer(planes)  # Batch normalization
        self.downsample = downsample  # Downsample module
        self.stride = stride  # Stride value for the convolutional layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BasicBlock.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - out (Tensor): Output tensor.
        """
        identity = x

        out = self.conv1(x)  # First convolutional layer
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv2(out)  # Second convolutional layer
        out = self.bn2(out)  # Batch normalization

        if self.downsample is not None:
            identity = self.downsample(x)  # Downsample the input if needed

        out += identity  # Add the residual connection
        out = self.relu(out)  # ReLU activation

        return out


# module to define a Bottleneck residual block for the resnet model
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution (self.conv2)
    # while the original implementation places the stride at the first 1x1 convolution (self.conv1)
    # according to "Deep residual learning for image recognition" (https://arxiv.org/abs/1512.03385).
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Bottleneck residual block implementation used in ResNet.

        Args:
        - inplanes (int): Number of input channels.
        - planes (int): Number of output channels.
        - stride (int): Stride value for the convolutional layers (default: 1).
        - downsample (nn.Module, optional): Downsample module (default: None).
        - groups (int): Number of groups for grouped convolution (default: 1).
        - base_width (int): Base width for grouped convolution (default: 64).
        - dilation (int): Dilation rate for dilated convolution (default: 1).
        - norm_layer (Callable[..., nn.Module], optional): Normalization layer (default: nn.BatchNorm2d).
        """
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)  # 1x1 convolutional layer
        self.bn1 = norm_layer(width)  # Batch normalization
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # 3x3 convolutional layer
        self.bn2 = norm_layer(width)  # Batch normalization
        self.conv3 = conv1x1(width, planes * self.expansion)  # 1x1 convolutional layer
        self.bn3 = norm_layer(planes * self.expansion)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation function
        self.downsample = downsample  # Downsample module
        self.stride = stride  # Stride value for the convolutional layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Bottleneck.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - out (Tensor): Output tensor.
        """
        identity = x

        out = self.conv1(x)  # First 1x1 convolutional layer
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv2(out)  # 3x3 convolutional layer
        out = self.bn2(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv3(out)  # Second 1x1 convolutional layer
        out = self.bn3(out)  # Batch normalization

        if self.downsample is not None:
            identity = self.downsample(x)  # Downsample the input

        out += identity  # Residual connection
        out = self.relu(out)  # ReLU activation

        return out

# module defining modified RESNET backbone
class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 5,
            img_size: int = 64,
            metadata_size: int = 32,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
        ) -> None:
        super(ResNet, self).__init__()

        # If norm_layer is not provided, default to nn.BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        self.img_size = img_size
        self.metadata_size = metadata_size

        # Check if replace_stride_with_dilation is provided
        if replace_stride_with_dilation is None:
            # If not provided, set it to a default value of [False, False, False]
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers (layer1, layer2, layer3, layer4)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Adaptive average pooling and fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_meta = nn.Linear(3, metadata_size)
        self.fc_img = nn.Linear(512 * block.expansion, img_size)

        self.fc_combined = nn.Linear(metadata_size +img_size,num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.soft = nn.Softmax(dim = 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # Adjust dilation and stride if dilate is True
        if dilate:
            self.dilation *= stride
            stride = 1

        # Create downsample layer if stride != 1 or number of input channels is different from output channels
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Add the first block of the layer with potential downsampling
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion

        # Add the rest of the blocks in the layer
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, metadata: Tensor) -> Tensor:

        metadata = F.relu(self.fc_meta(metadata))
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # First residual layer
        x = self.layer2(x)  # Second residual layer
        x = self.layer3(x)  # Third residual layer
        x = self.layer4(x)  # Fourth residual layer

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_img(x)

        total_length =  self.img_size + self.metadata_size

        combined_features = torch.cat((x, metadata), dim=1)
        combined_features = self.dropout(combined_features)
        x = self.fc_combined(combined_features)
        return x

    def forward(self, x: Tensor, metadata: Tensor) -> Tensor:
        """
        Forward pass of the ResNet model.

        Args:
            x (Tensor): Input image tensor.
            metadata (Tensor): Metadata tensor.
            spectral_data (Tensor): Spectral data tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._forward_impl(x, metadata)


# function to train the revised Resnet model
def resnet_merge(block: Type[Union[BasicBlock, Bottleneck]],
               layers: List[int],
               pretrained: bool = False,
               num_classes: int = 5,
               metadata_size: int = 32,
               img_size: int = 64,
               progress: bool = True,
               **kwargs: Any) -> ResNet:
    """
    Create a new ResNet model.

    Args:
        block (Type[Union[BasicBlock, Bottleneck]]): Type of the residual block (BasicBlock or Bottleneck).
        layers (List[int]): List specifying the number of blocks in each layer of the network.
        pretrained (bool): Whether to load a pretrained ResNet model. Default is False.
        num_classes (int): Number of output classes. Default is 17.
        metadata_size (int): Size of the metadata input. Default is 32.
        img_size (int): Size of the image input. Default is 64.
        spectral_size (int): Size of the spectral data input. Default is 32.
        progress (bool): Whether to display a progress bar when downloading pretrained weights. Default is True.
        **kwargs (Any): Additional keyword arguments to pass to the ResNet constructor.

    Returns:
        ResNet: ResNet model.
    """
    if pretrained:
        # Load a pretrained ResNet18 model
        model = resnet.resnet18(pretrained=True, progress=progress)
        # Update the final fully connected layer for the desired number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Update the kwargs dictionary with the specified parameters
    kwargs['metadata_size'] = metadata_size
    kwargs['img_size'] = img_size
    kwargs['block'] = block
    kwargs['layers'] = layers

    # Create a new ResNet model with modified parameters
    model = ResNet(**kwargs)
    return model
        

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
test_dataset = FishTestDataset( data_dir,config["test_csv"], data_transforms)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet_merge(
    BasicBlock,
    [2, 2, 2, 2],
    num_classes=5,
    pretrained=False,
    metadata_size=32,
    img_size=64,
)

# Load model - TODO
model.load_state_dict(torch.load(config["model_path"]))

model.eval()    
model.to(device)

import torch

output_path = "inference_results.csv"
file = open(config["out_path"], 'w')
file.write("Image Name, Predicted Age\n")

for (images, meta), img_path in tqdm(test_loader):
    images = images.to(device)
    meta = meta.to(device)
    outputs = model(images, meta)
    outputs = torch.squeeze(outputs)
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().detach().numpy()
    for i in range(preds.shape[0]):
        age = str(preds[i])
        if(preds[i] ==4):
            age = "4+"
        file.write("%s,%s\n"%(img_path[i],age))
file.close()
