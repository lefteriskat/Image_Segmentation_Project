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

from src.models.utils import plot_predictions, prediction_accuracy


def bce_loss(y_real, y_pred):
    y_pred = torch.clip(y_pred, -10, 10)
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
    epochs = 20
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
                    A.RandomCrop(width=resize_dims, height=resize_dims, p=0.5),
                ],
                p=1.0,
            ),
            A.Transpose(p),
            A.OneOf(
                [
                    A.ToGray(p=0.2),
                    A.RandomGamma(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.ElasticTransform(p=0.4),
                ],
                p=0.5,
            ),
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

    # train_loader, validation_loader, test_loader = universal_dataloader.getDataLoader(
    #     "ph", train_transform, val_transform, test_transform, batch_size=batch_size
    # )

    train_dataset, validation_dataset, test_dataset = universal_dataloader.get_datasets(
        "ph", train_transform, val_transform, test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model instanciating
    # model = EncDecModel(3, 1, 64)
    model = UNetBlocked(in_channels=3, out_channels=1, unet_block="resnet")
    # model = UNet()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 150, 300, 350], gamma=0.5
    )

    # Loss function
    loss_func = bce_loss
    # loss_func = dice_loss
    # loss_func = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):
        train_avg_loss = 0
        train_accuracy = 0
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
            pred_sigmoided = F.sigmoid(pred)

            loss = loss_func(masks_batch, pred_sigmoided)  # forward-pass
            loss.backward()  # backward-pass
            optimizer.step()  # update weights

            # calculate metrics to show the user
            train_avg_loss += loss / len(train_loader)
            train_accuracy += prediction_accuracy(masks_batch, pred_sigmoided) / (
                len(train_dataset) * resize_dims**2
            )

        print(
            f" - Training loss: {train_avg_loss}  - Training accuracy: {train_accuracy}"
        )

        # Compute the evaluation set loss
        validation_avg_loss = 0
        model.eval()
        for images_batch, masks_batch in tqdm(
            validation_loader, desc="Validation", leave=None
        ):
            masks_batch = masks_batch.float().unsqueeze(1)
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            with torch.no_grad():
                pred = model(images_batch)

            loss = loss_func(masks_batch, pred)

            validation_avg_loss += loss / len(validation_dataset)

        print(" - Validation loss: %f" % validation_avg_loss)

        # Adjust lr
        # scheduler.step()

    # Test results and plot
    test_avg_loss = 0
    for images_batch, masks_batch in tqdm(test_loader, desc="Test"):
        masks_batch = masks_batch.float().unsqueeze(1)
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
        with torch.no_grad():
            pred = model(images_batch)

        test_avg_loss += loss_func(masks_batch, pred) / len(test_dataset)

    print(" - Test loss: %f" % test_avg_loss)

    plot_predictions(images_batch, masks_batch, pred)


if __name__ == "__main__":
    main()
