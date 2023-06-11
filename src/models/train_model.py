import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.ops as ops

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

import src.models.utils as utils


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

def focal_loss(y_real, y_pred):
    fc = ops.sigmoid_focal_loss(y_real, y_pred) # alpha: float = 0.25, gamma: float = 2, reduction: str = 'none')
    return fc

def cross_entropy(y_real, y_pred, weighted=False, weights=[]):
    if weighted==False:

        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_real, y_pred)
    else:
        weight_tensor = torch.tensor(weights)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        loss = criterion(y_real, y_pred)
    return loss
    

def main():
    # Paths and constants
    ph2_data_path = "/dtu/datasets1/02514/PH2_Dataset_images"

    dataset_name = 'ph' # 'drive' or 'ph'

    seed = 7

    resize_dims = 128
    batch_size = 16  # we do not have many images
    epochs = 250
    n_epochs_save = 10  # save every n_epochs_save epochs
    lr = 1e-3

    # Names and other identifiers
    model_name = "vgg"
    

    torch.manual_seed(seed)

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
                    A.RandomSizedCrop(
                        min_max_height=[resize_dims / 4, resize_dims / 2],
                        width=resize_dims,
                        height=resize_dims,
                        p=0.5,
                    ),
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


    #### CHANGE THIS!! FOR BASELINE ONLY!!!!
    train_transform = val_transform

    # train_loader, validation_loader, test_loader = universal_dataloader.getDataLoader(
    #     "ph", train_transform, val_transform, test_transform, batch_size=batch_size
    # )

    train_dataset, validation_dataset, test_dataset = universal_dataloader.get_datasets(
        dataset_name, train_transform, val_transform, test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Model instanciating
    # model = EncDecModel(3, 1, 64)
    model = UNetBlocked(in_channels=3, out_channels=1, unet_block=model_name)
    # model = UNet()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs // 4, epochs // 2, int(epochs * 0.75)], gamma=0.5
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
        tp, tn, fp, fn = [0]*4
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

            loss = loss_func(masks_batch, pred)  # forward-pass
            loss.backward()  # backward-pass
            optimizer.step()  # update weights

            # calculate metrics to show the user
            train_avg_loss += loss / len(train_loader)
            train_accuracy += utils.prediction_accuracy(masks_batch, pred_sigmoided) / (
                len(train_dataset) * resize_dims**2
            )

            tp_, tn_, fp_, fn_ = utils.get_tp_tn_fp_fn(masks_batch, pred_sigmoided)
            
            tp += tp_
            fp += fp_
            tn += tn_
            fn += fn_

        test_sens, test_spec = utils.get_sensitivity_specificity(tp,tn, fp, fn)
        test_dice = utils.get_dice_coe(masks_batch, pred_sigmoided)
        test_iou = utils.get_IoU(tp, fp, fn)

        print(
            f" - Training loss: {train_avg_loss}  - Training accuracy: {train_accuracy}"
        )
        print(f" - Train sensitivity: {test_sens}  - Train specificity: {test_spec}")
        print(f" - Train DICE: {test_dice}  - Train IoU: {test_iou}")

        # Compute the evaluation set loss
        validation_avg_loss = 0
        validation_accuracy = 0
        tp, tn, fp, fn = [0]*4
        model.eval()
        for images_batch, masks_batch in tqdm(
            validation_loader, desc="Validation", leave=None
        ):
            masks_batch = masks_batch.float().unsqueeze(1)
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            with torch.no_grad():
                pred = model(images_batch)
                pred_sigmoided = F.sigmoid(pred)

            # loss = loss_func(masks_batch, pred)
            loss = loss_func(masks_batch, pred)

            validation_avg_loss += loss / len(validation_dataset)
            validation_accuracy += utils.prediction_accuracy(masks_batch, pred_sigmoided) / (
                len(validation_dataset) * resize_dims**2
            )

            tp_, tn_, fp_, fn_ = utils.get_tp_tn_fp_fn(masks_batch, pred_sigmoided)
            
            tp += tp_
            fp += fp_
            tn += tn_
            fn += fn_

        test_sens, test_spec = utils.get_sensitivity_specificity(tp,tn, fp, fn)
        test_dice = utils.get_dice_coe(masks_batch, pred_sigmoided)
        test_iou = utils.get_IoU(tp, fp, fn)

        # print(" - Validation loss: %f" % validation_avg_loss)
        print(
            f" - Validation loss: {validation_avg_loss}  - Validation accuracy: {validation_accuracy}"
        )
        print(f" - Validation sensitivity: {test_sens}  - Validation specificity: {test_spec}")
        print(f" - Validation DICE: {test_dice}  - Validation IoU: {test_iou}")
        # Adjust lr
        scheduler.step()

    # Test results and plot
    test_avg_loss = 0
    test_accuracy = 0
    tp, tn, fp, fn = [0]*4
    for images_batch, masks_batch in tqdm(test_loader, desc="Test"):
        masks_batch = masks_batch.float().unsqueeze(1)
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
        with torch.no_grad():
            pred = model(images_batch)
            pred_sigmoided = F.sigmoid(pred)

        test_avg_loss += loss_func(masks_batch, pred) / len(test_dataset)
        test_accuracy += utils.prediction_accuracy(masks_batch, pred_sigmoided) / (
            len(test_dataset) * resize_dims**2
        )

        tp_, tn_, fp_, fn_ = utils.get_tp_tn_fp_fn(masks_batch, pred_sigmoided)
            
        tp += tp_
        fp += fp_
        tn += tn_
        fn += fn_

    test_sens, test_spec = utils.get_sensitivity_specificity(tp,tn, fp, fn)
    test_dice = utils.get_dice_coe(masks_batch, pred_sigmoided)
    test_iou = utils.get_IoU(tp, fp, fn)
    # print(" - Test loss: %f" % test_avg_loss)
    print(f" - Test loss: {test_avg_loss}  - Test accuracy: {test_accuracy}")
    print(f" - Test sensitivity: {test_sens}  - Test specificity: {test_spec}")
    print(f" - Test DICE: {test_dice}  - Test IoU: {test_iou}")

    utils.plot_predictions(images_batch, masks_batch, pred)


if __name__ == "__main__":
    main()
