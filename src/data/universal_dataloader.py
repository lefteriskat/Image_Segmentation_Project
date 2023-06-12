from src.data.DRIVE_dataloader import DRIVEDataset
from src.data.ph_dataset import PH2Dataset
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.image

from PIL import Image

from src.data.make_dataset import EyeVesselsTest

def getDataLoader(
    dataset: str,
    train_transforms,
    val_transforms,
    test_transforms,
    batch_size=4,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
):
    target_train_dataset: torch.utils.data.Dataset = None
    target_val_dataset: torch.utils.data.Dataset = None
    target_test_dataset: torch.utils.data.Dataset = None
    if dataset == "drive":
        target_train_dataset = DRIVEDataset(
            root_dir="/dtu/datasets1/02514/DRIVE/training", transform=train_transforms
        )
        target_val_dataset = DRIVEDataset(
            root_dir="/dtu/datasets1/02514/DRIVE/training", transform=val_transforms
        )
        target_test_dataset = DRIVEDataset(
            root_dir="/dtu/datasets1/02514/DRIVE/training", transform=test_transforms
        )
    elif dataset == "ph":
        target_train_dataset = PH2Dataset(
            root_dir="/dtu/datasets1/02514/PH2_Dataset_images",
            transform=train_transforms,
        )
        target_val_dataset = PH2Dataset(
            root_dir="/dtu/datasets1/02514/PH2_Dataset_images", transform=val_transforms
        )
        target_test_dataset = PH2Dataset(
            root_dir="/dtu/datasets1/02514/PH2_Dataset_images",
            transform=test_transforms,
        )

    print(f"Total length: {len(target_train_dataset)}")

    indices = torch.randperm(len(target_train_dataset))
    val_size = int(np.floor(len(target_train_dataset) * val_fraction))
    test_size = int(np.floor(len(target_train_dataset) * test_fraction))

    train_dataset = torch.utils.data.Subset(
        target_train_dataset, indices[: -(val_size + test_size)]
    )
    val_dataset = torch.utils.data.Subset(target_val_dataset, indices[-val_size:])
    test_dataset = torch.utils.data.Subset(
        target_test_dataset, indices[-(val_size + test_size) : -val_size]
    )

    print(f"Train length: {len(train_dataset)}")
    print(f"Validation length: {len(val_dataset)}")
    print(f"Test length: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader, val_loader, test_loader


def get_datasets(
    dataset: str,
    train_transforms,
    val_transforms,
    test_transforms,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
):
    target_train_dataset: torch.utils.data.Dataset = None
    target_val_dataset: torch.utils.data.Dataset = None
    target_test_dataset: torch.utils.data.Dataset = None
    if dataset == "drive":
        target_train_dataset = DRIVEDataset(
            root_dir="/dtu/datasets1/02514/DRIVE/training", transform=train_transforms
        )
        target_val_dataset = DRIVEDataset(
            root_dir="/dtu/datasets1/02514/DRIVE/training", transform=val_transforms
        )
        target_test_dataset = DRIVEDataset(
            root_dir="/dtu/datasets1/02514/DRIVE/training", transform=test_transforms
        )
    elif dataset == "ph":
        target_train_dataset = PH2Dataset(
            root_dir="/dtu/datasets1/02514/PH2_Dataset_images",
            transform=train_transforms,
        )
        target_val_dataset = PH2Dataset(
            root_dir="/dtu/datasets1/02514/PH2_Dataset_images", transform=val_transforms
        )
        target_test_dataset = PH2Dataset(
            root_dir="/dtu/datasets1/02514/PH2_Dataset_images",
            transform=test_transforms,
        )
    
    print(f"Total length: {len(target_train_dataset)}")

    indices = torch.randperm(len(target_train_dataset))
    val_size = int(np.floor(len(target_train_dataset) * val_fraction))
    test_size = int(np.floor(len(target_train_dataset) * test_fraction))

    train_dataset = torch.utils.data.Subset(
        target_train_dataset, indices[: -(val_size + test_size)]
    )
    val_dataset = torch.utils.data.Subset(target_val_dataset, indices[-val_size:])
    test_dataset = torch.utils.data.Subset(
        target_test_dataset, indices[-(val_size + test_size) : -val_size]
    )

    print(f"Train length: {len(train_dataset)}")
    print(f"Validation length: {len(val_dataset)}")
    print(f"Test length: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def create_test_image_dir(
    test_dataset_name: str, resize_dims: int = 128, n_images: int = 10
) -> None:
    
    read_root_dir_name = '/dtu/datasets1/02514/PH2_Dataset_images'
    if test_dataset_name=='drive':
        read_root_dir_name = '/dtu/datasets1/02514/DRIVE'

    save_root_dir_path = f"test_images_{test_dataset_name}"
    path_ = Path(save_root_dir_path)

    path_.mkdir(exist_ok=True)

    transforms = A.Compose(
        [
            A.Resize(width=resize_dims, height=resize_dims),
            ToTensorV2(),
        ]
    )

    _, _, test_dataset = get_datasets(
        test_dataset_name, transforms, transforms, transforms
    )
    if test_dataset_name=='drive':
        test_dataset = EyeVesselsTest(read_root_dir_name,train=False, transform=transforms)


    dataset_len = len(test_dataset)

    if dataset_len < n_images:
        n_images = dataset_len

    print(test_dataset_name)

    for i in range(n_images):
        image, _ = test_dataset[i]


        image = image.permute(1, 2, 0).numpy()

        # im = Image.fromarray(image)
        # im.save(images_path.as_posix() + f"/{test_dataset_name}_{i}.png", "PNG")

        matplotlib.image.imsave(
            path_.as_posix() + f"/{test_dataset_name}_{i}.png", image
        )

        



def main():
    # transform = transforms.Compose(
    #     [transforms.Resize((256, 256)), transforms.ToTensor()]
    # )
    # train, val, test = getDataLoader("drive", transform, transform, transform)
    # train, val, test = getDataLoader("ph", transform, transform, transform)

    # # print(len(train))

    # a, b = next(iter(train))

    # print(a.shape, b.shape)
    # print(a.dtype, b.dtype)

    resize_dims = 256

    create_test_image_dir("ph", n_images=10, resize_dims=resize_dims)

    create_test_image_dir("drive", n_images=10, resize_dims=resize_dims)


if __name__ == "__main__":
    main()
