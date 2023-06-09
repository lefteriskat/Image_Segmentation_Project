import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.data.ph_dataset import PH2Dataset
from src.models.models import EncDecModel

def bce_loss(y_real, y_pred):
    y_pred = torch.clip(y_pred, -10, 10)
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def main():
    # Paths and constants
    ph2_data_path = '/dtu/datasets1/02514/PH2_Dataset_images'

    resize_dims = 128
    batch_size = 4 # we do not have many images
    epochs = 40

    # Device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Runs on: {device}")

    # Datasets and data loaders
    
    train_transform = transforms.Compose([transforms.Resize((resize_dims, resize_dims)),
                                          transforms.ToTensor()])
    
    test_transform = train_transform

    train_dataset = PH2Dataset(root_dir=ph2_data_path, transform=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True
                              )
    
    eval_dataset = PH2Dataset(root_dir=ph2_data_path, transform=test_transform)
    eval_loader = DataLoader(eval_dataset,
                              batch_size=batch_size,
                              shuffle=False
                              )
    

    # Model instanciating
    model = EncDecModel(3, 1, 64)
    model.to(device)

    # Optimizer
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # Loss function
    loss_func = bce_loss

    # Training loop
    for epoch in tqdm(range(epochs), desc='Epoch'):

        train_avg_loss = 0
        model.train()  # train mode
        
        for images_batch, masks_batch in tqdm(train_loader, leave=None, desc='Training'):
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            pred = model(images_batch)
            loss = loss_func(masks_batch, pred)  # forward-pass
            loss.backward()  # backward-pass
            optimizer.step()  # update weights

            # calculate metrics to show the user
            train_avg_loss += loss / len(train_loader)

        print(' - Training loss: %f' % train_avg_loss)

        #Compute the evaluation set loss
        eval_avg_loss = 0
        model.eval()
        for images_batch, masks_batch in tqdm(eval_loader, desc='Eval', leave=None):
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            with torch.no_grad():
                pred = model(images_batch)

            loss = loss_func(masks_batch, pred)

            eval_avg_loss+= loss / len(eval_loader)
        
        print(' - Eval loss: %f' % eval_avg_loss)


if __name__=='__main__':
    main()

