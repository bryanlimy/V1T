import torch
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
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), dpi=args.dpi)
        image, output = images[i][0], outputs[i][0]
        axes[0].imshow(images, cmap=tensorboard.GRAY, vmin=0, vmax=1, aspect="auto")
        axes[1].imshow(outputs, cmap=tensorboard.GRAY, vmin=0, vmax=1, aspect="auto")
        axes.set_title(f"MSE: {((image - output)**2).mean():.04f}", pad=3, fontsize=10)
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
                "loss/loss": loss.detach(),
                "loss/reg_loss": reg_loss.detach(),
                "loss/total_loss": total_loss.detach(),
            },
        )
    for k, v in results.items():
        results[k] = torch.stack(v).mean()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results


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
        for images, _ in tqdm(ds, desc="Train", disable=args.verbose == 0):
            images = images.to(model.device)
            outputs = model(images)
            loss = F.mse_loss(input=outputs, target=images)
            reg_loss = model.regularizer()
            total_loss = loss + args.reg_scale * reg_loss
            utils.update_dict(
                results,
                {
                    "loss/loss": loss.detach(),
                    "loss/reg_loss": reg_loss.detach(),
                    "loss/total_loss": total_loss.detach(),
                },
            )
            if make_plot:
                plot_image(
                    args,
                    images=images.cpu() * data.IMAGE_STD + data.IMAGE_MEAN,
                    outputs=outputs.cpu() * data.IMAGE_STD + data.IMAGE_MEAN,
                    summary=summary,
                    epoch=epoch,
                    mode=mode,
                )
    for k, v in results.items():
        results[k] = torch.stack(v).mean()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results
