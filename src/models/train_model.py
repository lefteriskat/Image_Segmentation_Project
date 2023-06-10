import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import torchvision.transforms as transforms
import albumentations as A
import albumentations.augmentations as augm
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.data.ph_dataset import PH2Dataset
from src.models.models import EncDecModel, UNet, UNetBlocked

from src.data import universal_dataloader

import src.data.utils as metrics

from src.models.utils import plot_predictions


def bce_loss(y_real, y_pred):
    y_pred = torch.clip(y_pred, -100, 100)
    return torch.mean(y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred)))


def dice_loss(y_real, y_pred):
    ### y_real and y_pred is [batch_n, channels=1, h, w]: [6, 1, 128, 128]
    y_real_flat = y_real.view(y_real.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    num = (2 * y_real_flat * F.sigmoid(y_pred_flat) + 1).mean()
    den = (y_real_flat + F.sigmoid(y_pred_flat)).mean() + 1
    return 1 - (num / den)


def main():
    # Paths and constants
    ph2_data_path = "/dtu/datasets1/02514/PH2_Dataset_images"

    resize_dims = 128
    batch_size = 16  # we do not have many images
    epochs = 80
    n_epochs_save = 10  # save every n_epochs_save epochs
    lr = 1e-4

    # Names and other identifiers
    model_name = "baseline"

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    # Datasets and data loaders

    # train_transform = transforms.Compose([transforms.Resize((resize_dims, resize_dims)),
    #                                       transforms.ToTensor()])

    # Define the training augmentations for the training data
    p = 0.5
    train_transform = A.Compose(
        [
            A.OneOf(
                [
                    A.Resize(width=resize_dims, height=resize_dims, p=0.5),
                    A.RandomCrop(width=resize_dims, height=resize_dims),
                ],
                p=1.0,
            ),
            A.Transpose(p),
            A.GaussianBlur(p=0.3),
            A.Blur(p=0.2),
            A.HorizontalFlip(p),
            A.VerticalFlip(p),
            A.Rotate(90, border_mode=cv2.BORDER_CONSTANT, p=p),
            A.Normalize([0.0] * 3, [1.0] * 3, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(width=resize_dims, height=resize_dims),
            A.Normalize([0.0] * 3, [1.0] * 3, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    test_transform = val_transform

    train_loader, eval_loader, test_loader = universal_dataloader.getDataLoader(
        "ph", train_transform, val_transform, test_transform, batch_size=batch_size
    )

    # Model instanciating
    # model = EncDecModel(3, 1, 64)
    # model = UNetBlocked(in_channels=3, out_channels=1, unet_block="resnet")
    model = UNet()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 150, 300, 350], gamma=0.5
    )

    # Loss function
    # loss_func = bce_loss
    loss_func = dice_loss
    # loss_func = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):
        train_avg_loss = 0
        model.train()  # train mode
        train_dice_score = 0
        for images_batch, masks_batch in tqdm(
            train_loader, leave=None, desc="Training"
        ):
            masks_batch = masks_batch.float().unsqueeze(1)

            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            pred = model(images_batch)

            # pred = F.sigmoid(pred)
            # pred = torch.where(pred>0.5, 1, 0)

            loss = loss_func(masks_batch, pred)  # forward-pass
            loss.backward()  # backward-pass
            optimizer.step()  # update weights

            # calculate metrics to show the user
            train_avg_loss += loss / len(train_loader)
            Y_hat = F.sigmoid(pred).detach().cpu()
            # print(metrics.compute_dice(Y_hat.cpu().type(torch.int64), masks_batch.cpu().type(torch.int64)))

        print(" - Training loss: %f" % train_avg_loss)

        # if epoch%n_epochs_save==0:
        #     torch.save()

        # Compute the evaluation set loss
        eval_avg_loss = 0
        model.eval()
        for images_batch, masks_batch in tqdm(eval_loader, desc="Eval", leave=None):
            masks_batch = masks_batch.float().unsqueeze(1)
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            with torch.no_grad():
                pred = model(images_batch)

            loss = loss_func(masks_batch, pred)

            eval_avg_loss += loss / len(eval_loader)

        print(" - Eval loss: %f" % eval_avg_loss)

        # Adjust lr
        # scheduler.step()

    # Test results and plot
    test_avg_loss = 0
    for images_batch, masks_batch in tqdm(test_loader, desc="Test"):
        masks_batch = masks_batch.float().unsqueeze(1)
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
        with torch.no_grad():
            pred = model(images_batch)

        test_avg_loss += loss_func(masks_batch, pred) / len(test_loader)

    print(" - Test loss: %f" % test_avg_loss)

    plot_predictions(images_batch, masks_batch, pred)


if __name__ == "__main__":
    main()
