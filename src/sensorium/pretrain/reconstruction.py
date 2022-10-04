import torch
import numpy as np
import typing as t
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sensorium.pretrain import data
from sensorium.utils import utils, tensorboard


def plot_image(
    args,
    images: torch.Tensor,
    outputs: torch.Tensor,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
    num_plots: int = 5,
):
    for i in range(min(num_plots, len(images))):
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), dpi=args.dpi)
        image, output = images[i][0], outputs[i][0]
        axes[0].imshow(image, cmap=tensorboard.GRAY, vmin=0, vmax=1, aspect="auto")
        axes[1].imshow(output, cmap=tensorboard.GRAY, vmin=0, vmax=1, aspect="auto")
        figure.suptitle(f"MSE: {((image - output)**2).mean():.04f}", fontsize=10)
        summary.figure(f"images/image{i:03d}", figure=figure, step=epoch, mode=mode)


def train(
    args,
    ds: DataLoader,
    model: nn.Module,
    optimizer: torch.optim,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {}
    model.train(True)
    for images, _ in tqdm(ds, desc="Train", disable=args.verbose == 0):
        images = images.to(model.device)
        outputs = model(images)
        loss = F.mse_loss(input=outputs, target=images)
        reg_loss = model.regularizer()
        total_loss = loss + args.reg_scale * reg_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        utils.update_dict(
            results,
            {
                "loss/loss": loss.item(),
                "loss/reg_loss": reg_loss.item(),
                "loss/total_loss": total_loss.item(),
            },
        )
        del loss, reg_loss, total_loss, outputs
    for k, v in results.items():
        results[k] = np.mean(v)
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results


def transform(image: torch.Tensor):
    """reverse image standardization"""
    return image.cpu() * data.IMAGE_STD + data.IMAGE_MEAN


def validate(
    args,
    ds: DataLoader,
    model: nn.Module,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    results, make_plot = {}, True
    model.train(False)
    with torch.no_grad():
        for images, _ in tqdm(ds, desc="Val", disable=args.verbose == 0):
            images = images.to(model.device)
            outputs = model(images)
            loss = F.mse_loss(input=outputs, target=images)
            reg_loss = model.regularizer()
            total_loss = loss + args.reg_scale * reg_loss
            utils.update_dict(
                results,
                {
                    "loss/loss": loss.item(),
                    "loss/reg_loss": reg_loss.item(),
                    "loss/total_loss": total_loss.item(),
                },
            )
            if make_plot:
                plot_image(
                    args,
                    images=data.reverse(images),
                    outputs=data.reverse(outputs),
                    summary=summary,
                    epoch=epoch,
                    mode=mode,
                )
                make_plot = False
            del loss, reg_loss, total_loss, outputs
    for k, v in results.items():
        results[k] = np.mean(v)
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results
