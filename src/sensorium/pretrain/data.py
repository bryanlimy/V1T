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
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.RandomCrop(size=(IMAGE_SIZE[1:]), pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomAdjustSharpness(sharpness_factor=0.6, p=0.25),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.RandomCrop(size=(IMAGE_SIZE[1:]), pad_if_needed=True),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )

    train_data = ImageFolder(root=data_dir, transform=train_transforms)
    val_data = ImageFolder(root=data_dir, transform=test_transforms)
    test_data = ImageFolder(root=data_dir, transform=test_transforms)

    size = len(train_data)
    indexes = np.arange(size)
    np.random.shuffle(indexes)

    size = 200
    indexes = indexes[:size]

    train_idx = indexes[: int(size * 0.7)]
    val_idx = indexes[int(size * 0.7) : int(size * 0.85)]
    test_idx = indexes[int(size * 0.85) :]

    # settings for DataLoader
    dataloader_kwargs = {"batch_size": batch_size, "num_workers": args.num_workers}
    if device.type in ["cuda", "mps"]:
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)

    train_ds = data.DataLoader(
        train_data, sampler=data.SubsetRandomSampler(train_idx), **dataloader_kwargs
    )
    val_ds = data.DataLoader(
        val_data, sampler=data.SubsetRandomSampler(val_idx), **dataloader_kwargs
    )
    test_ds = data.DataLoader(
        test_data, sampler=data.SubsetRandomSampler(test_idx), **dataloader_kwargs
    )

    args.input_shape = IMAGE_SIZE
    if args.mode == 0:
        args.output_shape = (NUM_CLASSES,)
    else:
        args.output_shape = args.input_shape

    return train_ds, val_ds, test_ds
