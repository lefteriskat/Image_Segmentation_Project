#123
import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt

# Use the Albumentations library for image and mask augmentation, to avoid
# having to handle the seed of torchvision.transforms
# and always apply the same transform to the mask and image
# https://albumentations.ai/docs/getting_started/mask_augmentation/
import albumentations as A
from albumentations.pytorch import ToTensorV2


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

        image_np = np.array(image)/255.
        lesion_np = np.array(lesion)

    
        # Perform the transformations to the image and the mask
        augmented = self.transform(image=image_np, mask=lesion_np)

        # Transform the data
        # X = self.transform(image)
        # Y = self.transform(lesion)

        X = augmented['image']
        Y = augmented['mask']
        
        return X, Y



def main():
    # The same transform can be used for both train and test, assuming you want to resize and convert to tensor
    # transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(),
        # A.Normalize()
        ])

    # Create Datasets and DataLoaders
    trainset = PH2Dataset(root_dir='/dtu/datasets1/02514/PH2_Dataset_images', transform=transform)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    print("MIA XARAAAA")
    a=next(iter(train_loader))
    print(a[0].shape, a[1].shape)
    print(a[0].max(), a[1].max())

if __name__=='__main__':
    main()

#testset = PH2Dataset(root_dir='./PH2_Dataset_image/test', transform=transform)
#test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
