import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.data.ph_dataset import PH2Dataset
from src.models.models import EncDecModel, UNet, UNetBlocked

from src.data import universal_dataloader

import src.data.utils as metrics 

def bce_loss(y_real, y_pred):
    y_pred = torch.clip(y_pred, -10, 10)
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def main():
    # Paths and constants
    ph2_data_path = '/dtu/datasets1/02514/PH2_Dataset_images'

    resize_dims = 128
    batch_size = 4 # we do not have many images
    epochs = 60
    n_epochs_save = 10 # save every n_epochs_save epochs
    lr = 5*1e-4
    # Names and other identifiers
    model_name='baseline'



    # Device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Runs on: {device}")

    # Datasets and data loaders
    
    train_transform = transforms.Compose([transforms.Resize((resize_dims, resize_dims)),
                                          transforms.ToTensor()])
    
    test_transform = train_transform
    val_transform = train_transform
    
    train_loader, eval_loader, test_loader = universal_dataloader.getDataLoader("ph", train_transform, val_transform, test_transform)
    

    # Model instanciating
    # model = EncDecModel(3, 1, 64)
    model = UNetBlocked(in_channels=3, out_channels=1)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # Loss function
    loss_func = bce_loss

    # Training loop
    for epoch in tqdm(range(epochs), desc='Epoch'):

        train_avg_loss = 0
        model.train()  # train mode
        train_dice_score = 0
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
            Y_hat = F.sigmoid(pred).detach().cpu()
            # print(metrics.compute_dice(Y_hat.cpu().type(torch.int64), masks_batch.cpu().type(torch.int64)))

        
        print(' - Training loss: %f' % train_avg_loss)
              
    

        # if epoch%n_epochs_save==0:
        #     torch.save()

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

