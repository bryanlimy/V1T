import torch
import typing as t
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder


IMAGE_SIZE = (1, 144, 256)
NUM_CLASSES = 1000
IMAGE_MEAN = torch.tensor(0.44531356896770125)
IMAGE_STD = torch.tensor(0.2692461874154524)


def reverse(image: torch.Tensor):
    """reverse image standardization"""
    return image * IMAGE_STD + IMAGE_MEAN


def get_ds(args, data_dir: str, batch_size: int, device: torch.device):
    image_ds = ImageFolder(
        root=data_dir,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize(size=(IMAGE_SIZE[1:])),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ]
        ),
    )

    size = len(image_ds)

    # split images into train-val-test with ratio of 70%-15%-15%
    train_ds, val_ds, test_ds = data.random_split(
        image_ds,
        lengths=[int(size * 0.7), int(size * 0.15), int(size * 0.15)],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # settings for DataLoader
    dataloader_kwargs = {"batch_size": batch_size, "num_workers": args.num_workers}
    if device.type in ["cuda", "mps"]:
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)

    train_ds = data.DataLoader(train_ds, shuffle=True, **dataloader_kwargs)
    val_ds = data.DataLoader(val_ds, **dataloader_kwargs)
    test_ds = data.DataLoader(test_ds, **dataloader_kwargs)

    args.input_shape = IMAGE_SIZE
    if args.mode == 0:
        args.output_shape = (NUM_CLASSES,)
    else:
        args.output_shape = args.input_shape

    return train_ds, val_ds, test_ds
