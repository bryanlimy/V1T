import torch
import typing as t
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
from functools import partial

NUM_CLASSES = 1000  # ImageNet classes
# ImageNet mean and standard deviation from Sensorium train set
IMAGE_MEAN = torch.tensor(113.52469635009766)
IMAGE_STD = torch.tensor(64.55815124511719)
IMAGE_SIZE = (1, 144, 256)  # Sensorium image dimension


def reverse(image: torch.Tensor):
    """reverse image standardization"""
    return image * IMAGE_STD + IMAGE_MEAN


def transform(image: Image, resize_image: int):
    image = F.to_grayscale(image)
    # ToTensor convert PIL image to range [0, 1]
    image = F.to_tensor(image)
    # convert image to range [0, 255] to match Sensorium images
    image = image * 255.0
    # convert images to (1, 144, 256)
    image = F.resize(image, size=list(IMAGE_SIZE[1:]), antialias=False)
    if resize_image == 1:
        image = F.resize(image, size=[36, 64], antialias=False)
    image = (image - IMAGE_MEAN) / IMAGE_STD
    return image


def get_ds(args, data_dir: str, batch_size: int, device: torch.device):
    image_ds = ImageFolder(
        root=data_dir,
        transform=partial(transform, resize_image=args.resize_image),
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

    args.input_shape = (1, 36, 64) if args.resize_image else IMAGE_SIZE
    if args.mode == 0:
        args.output_shape = (NUM_CLASSES,)
    else:
        args.output_shape = args.input_shape

    return train_ds, val_ds, test_ds
