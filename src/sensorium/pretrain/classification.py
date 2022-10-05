import torch
import numpy as np
import typing as t
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    return (y_pred == y_true).float().sum().item()


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
        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        predictions = torch.argmax(outputs, dim=1)
        utils.update_dict(
            results,
            {
                "loss/loss": loss.item(),
                "loss/reg_loss": reg_loss.item(),
                "loss/total_loss": total_loss.item(),
                "accuracy": num_correct(labels, predictions),
            },
        )
        del loss, reg_loss, total_loss, outputs, predictions
    for k, v in results.items():
        if k == "accuracy":
            results["accuracy"] = 100 * (np.sum(v) / len(ds.dataset))
        else:
            results[k] = np.mean(v)
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
            predictions = torch.argmax(outputs, dim=1)
            utils.update_dict(
                results,
                {
                    "loss/loss": loss.item(),
                    "accuracy": num_correct(labels, predictions),
                },
            )
            if make_plot:
                plot_image(
                    args,
                    images=data.reverse(images.cpu()),
                    predictions=predictions.cpu(),
                    labels=labels.cpu(),
                    summary=summary,
                    epoch=epoch,
                    mode=mode,
                )
                make_plot = False
            del loss, outputs, predictions
    for k, v in results.items():
        if k == "accuracy":
            results["accuracy"] = 100 * (np.sum(v) / len(ds.dataset))
        else:
            results[k] = np.mean(v)
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
    return results
