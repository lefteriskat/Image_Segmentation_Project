#123
import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time

import matplotlib.pyplot as plt
from IPython.display import clear_output


class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        'Initialization'
        self.transform = transform
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'images/*.tif')))  # replace '*.jpg' with actual file extension
        self.label_paths = sorted(glob.glob(os.path.join(root_dir, '1st_manual/*.gif')))  # replace '*.jpg' with actual file extension
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Get paths for the image and the manual segmentation for the given index
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Open the image and the manual segmentation
        image = Image.open(image_path)
        label = Image.open(label_path)
        
        # Transform the data
        X = self.transform(image)
        Y = self.transform(label)
        
        return X, Y

# The same transform can be used for both train and test, assuming you want to resize and convert to tensor
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Create Dataset and DataLoader
dataset = DRIVEDataset(root_dir='/dtu/datasets1/02514/DRIVE/training', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

a=next(iter(data_loader))
print(a)
