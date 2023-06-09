from src.data.DRIVE_dataloader import DRIVEDataset
from src.data.ph_dataset import PH2Dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np


def getDataLoader(dataset:str, train_transforms, val_transforms, test_transforms, batch_size=4, train_fraction:float=0.7, val_fraction:float=0.1, test_fraction:float=0.2):
    target_train_dataset:torch.utils.data.Dataset = None
    target_val_dataset:torch.utils.data.Dataset = None
    target_test_dataset:torch.utils.data.Dataset = None
    if dataset == "drive":
        target_train_dataset = DRIVEDataset(root_dir='/dtu/datasets1/02514/DRIVE/training', transform=train_transforms)
        target_val_dataset = DRIVEDataset(root_dir='/dtu/datasets1/02514/DRIVE/training', transform=val_transforms)
        target_test_dataset = DRIVEDataset(root_dir='/dtu/datasets1/02514/DRIVE/training', transform=test_transforms)
    elif dataset == "ph":
        target_train_dataset = PH2Dataset(root_dir='/dtu/datasets1/02514/PH2_Dataset_images', transform=train_transforms)
        target_val_dataset = PH2Dataset(root_dir='/dtu/datasets1/02514/PH2_Dataset_images', transform=val_transforms)
        target_test_dataset = PH2Dataset(root_dir='/dtu/datasets1/02514/PH2_Dataset_images', transform=test_transforms)

    print("f Total length: {len(target_train_dataset)}") 

    indices = torch.randperm(len(target_train_dataset))
    val_size = int(np.floor(len(target_train_dataset)*val_fraction))
    test_size = int(np.floor(len(target_train_dataset)*test_fraction))


    train_dataset = torch.utils.data.Subset(target_train_dataset, indices[:-(val_size+test_size)])
    val_dataset = torch.utils.data.Subset(target_val_dataset, indices[-val_size:])
    test_dataset = torch.utils.data.Subset(target_test_dataset, indices[-(val_size+test_size):-val_size])

    print(f"Train length: {len(train_dataset)}")
    print(f"Validation length: {len(val_dataset)}")
    print(f"Test length: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader


def main():
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train, val, test = getDataLoader("drive", transform, transform, transform)
    train, val, test = getDataLoader("ph", transform, transform, transform)

    #print(len(train))


if __name__=='__main__':
    main()