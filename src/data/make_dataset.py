# -*- coding: utf-8 -*-
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

from PIL import Image

import numpy as np

class SkinLesionDataset(Dataset):
    '''
    Datset class for the Image Lesion segmentation task.
    No division in training and test set, so only path is needed.
    Added a dummy train argument for consistency.
    ''' 
    def __init__(self,train:bool, transform,  data_path:str )-> None:
        super().__init__()

        self.transform = transform

        self.images_root_paths = list(Path(data_path).iterdir())


        
    def get_subfolders(self, root_dir):
        specific_images_dirs = list(Path(root_dir).iterdir())
        return specific_images_dirs

    def get_data(self, data_dir):
        data_dir = sorted(data_dir)[:2]
        data = []
        for data_type in data_dir:
            for file_ in sorted(data_type.glob('*.bmp')):
                data.append(file_)

        return data

    
    def __len__(self):
        return len(self.images_root_paths)
    
    def __getitem__(self, index):
        specific_img_dir = self.images_root_paths[index]
        img_root_dir = self.get_subfolders(specific_img_dir)
        image_path, mask_path = self.get_data(img_root_dir)

        img = Image.open(image_path.as_posix())
        mask = Image.open(mask_path.as_posix())

        return self.transform(img), self.transform(mask)


def main():
    
    
    data_path = '/dtu/datasets1/02514/PH2_Dataset_images/'

    transform_ = transforms.Compose([
        transforms.ToTensor()
    ])

    data_set = SkinLesionDataset(True,transform_, data_path)

    loader = DataLoader(data_set, batch_size=1)

    a = next(iter(loader))

    print(a[0].shape, a[1].shape)



if __name__ == '__main__':
    
    main()
