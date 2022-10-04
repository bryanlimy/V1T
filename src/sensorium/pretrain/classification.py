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
    predictions: torch.Tensor,
    labels: torch.Tensor,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
    num_plots: int = 5,
):
    for i in range(min(num_plots, len(images))):
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=args.dpi)
        ax.imshow(images[i][0], cmap=tensorboard.GRAY, vmin=0, vmax=1, aspect="auto")
        ax.set_title(
            f"label: {labels[i]}   prediction: {predictions[i]}", pad=3, fontsize=10
        )
        summary.figure(f"images/image{i:03d}", figure=figure, step=epoch, mode=mode)


def num_correct(y_true: torch.Tensor, y_pred: torch.Tensor):
    return (y_pred == y_true).float().sum()


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
    for images, labels in tqdm(ds, desc="Train", disable=args.verbose == 0):
        images, labels = images.to(model.device), labels.to(model.device)
        outputs = model(images)
        loss = F.nll_loss(input=outputs, target=labels)
        reg_loss = model.regularizer()
        total_loss = loss + args.reg_scale * reg_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        predictions = torch.argmax(outputs, dim=1)
        utils.update_dict(
            results,
            {
                "loss/loss": loss.detach(),
                "loss/reg_loss": reg_loss.detach(),
                "loss/total_loss": total_loss.detach(),
                "accuracy": num_correct(labels, predictions),
            },
        )
    for k, v in results.items():
        v = torch.stack(v)
        if k == "accuracy":
            results["accuracy"] = 100 * (v.sum() / len(ds.dataset))
        else:
            results[k] = v.mean()
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
        for images, labels in tqdm(ds, desc="Val", disable=args.verbose == 0):
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            loss = F.nll_loss(input=outputs, target=labels)
            reg_loss = model.regularizer()
            total_loss = loss + args.reg_scale * reg_loss
            predictions = torch.argmax(outputs, dim=1)
            utils.update_dict(
                results,
                {
                    "loss/loss": loss.detach(),
                    "loss/reg_loss": reg_loss.detach(),
                    "loss/total_loss": total_loss.detach(),
                    "accuracy": num_correct(labels, predictions),
                },
            )
            if make_plot:
                plot_image(
                    args,
                    images=images.cpu() * data.IMAGE_STD + data.IMAGE_MEAN,
                    predictions=predictions.cpu(),
                    labels=labels.cpu(),
                    summary=summary,
                    epoch=epoch,
                    mode=mode,
                )
                make_plot = False
    for k, v in results.items():
        v = torch.stack(v)
        if k == "accuracy":
            results["accuracy"] = 100 * (v.sum() / len(ds.dataset))
        else:
            results[k] = v.mean()
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
    return results
