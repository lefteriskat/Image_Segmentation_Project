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



class PH2Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        'Initialization'
        self.transform = transform
        self.root_dir = root_dir
        self.patient_folders = sorted(os.listdir(root_dir))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.patient_folders)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Get patient folder
        patient_folder = self.patient_folders[idx]

        # Create paths to the Dermoscopic image and lesion mask for the given patient
        image_path = os.path.join(self.root_dir, patient_folder, patient_folder+'_Dermoscopic_Image', patient_folder+'.bmp')  # replace 'image.jpg' with actual filename
        lesion_path = os.path.join(self.root_dir, patient_folder, patient_folder+'_lesion', patient_folder+'_lesion'+'.bmp')  # replace 'mask.jpg' with actual filename

        # Open the Dermoscopic image and the lesion mask
        image = Image.open(image_path)
        lesion = Image.open(lesion_path)
        
        # Transform the data
        X = self.transform(image)
        Y = self.transform(lesion)
        
        return X, Y



def main():
    # The same transform can be used for both train and test, assuming you want to resize and convert to tensor
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # Create Datasets and DataLoaders
    trainset = PH2Dataset(root_dir='/dtu/datasets1/02514/PH2_Dataset_images', transform=transform)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    print("MIA XARAAAA")
    a=next(iter(train_loader))
    print(a[0].shape, a[1].shape)

if __name__=='__main__':
    main()

#testset = PH2Dataset(root_dir='./PH2_Dataset_image/test', transform=transform)
#test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
