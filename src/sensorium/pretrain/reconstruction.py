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
        figure, axes = plt.subplots(
            nrows=1, ncols=2, gridspec_kw={"wspace": 0.2}, figsize=(10, 3), dpi=args.dpi
        )
        image, output = images[i][0], outputs[i][0]
        axes[0].imshow(image, cmap=tensorboard.GRAY, vmin=0, vmax=1, aspect="auto")
        axes[1].imshow(output, cmap=tensorboard.GRAY, vmin=0, vmax=1, aspect="auto")
        figure.suptitle(f"MSE: {((image - output)**2).mean():.04f}", fontsize=10)
        summary.figure(f"images/image{i:03d}", figure=figure, step=epoch, mode=mode)


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1-D Gaussian kernel with shape (1, 1, size)
    Args:
      size: the size of the Gaussian kernel
      sigma: sigma of normal distribution
    Returns:
      1D kernel (1, 1, size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(inputs: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Apply 1D Gaussian kernel to inputs images
    Args:
      inputs: a batch of images in shape (N,C,H,W)
      win: 1-D Gaussian kernel
    Returns:
      blurred images
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    channel = inputs.shape[1]
    outputs = inputs
    for i, s in enumerate(inputs.shape[2:]):
        if s >= win.shape[-1]:
            outputs = F.conv2d(
                outputs,
                weight=win.transpose(2 + i, -1),
                stride=1,
                padding=0,
                groups=channel,
            )
    return outputs


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    max_value: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """Computes structural similarity index metric (SSIM)

    Reference: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py

    Args:
      x: images in the format of (N,C,H,W)
      y: images in the format of (N,C,H,W)
      max_value: the maximum value of the images (usually 1.0 or 255.0)
      size_average: return SSIM average of all images
      win_size: the size of gauss kernel
      win_sigma: sigma of normal distribution
      win: 1-D gauss kernel. if None, a new kernel will be created according to
          win_size and win_sigma
      K1: scalar constant
      K2: scalar constant
    Returns:
      SSIM value(s)
    """
    assert x.shape == y.shape, "input images should have the same dimensions."

    # remove dimensions that has size 1, except the batch and channel dimensions
    for d in range(2, x.ndim):
        x = x.squeeze(dim=d)
        y = y.squeeze(dim=d)

    assert x.ndim == 4, f"input images should be 4D, but got {x.ndim}."
    assert win_size % 2 == 1, f"win_size should be odd, but got {win_size}."

    win = _gaussian_kernel_1d(win_size, win_sigma)
    win = win.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))

    compensation = 1.0

    C1 = (K1 * max_value) ** 2
    C2 = (K2 * max_value) ** 2

    win = win.to(x.device, dtype=x.dtype)

    mu1 = _gaussian_filter(x, win)
    mu2 = _gaussian_filter(y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (_gaussian_filter(x * x, win) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter(y * y, win) - mu2_sq)
    sigma12 = compensation * (_gaussian_filter(x * y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)

    return ssim_per_channel.mean() if size_average else ssim_per_channel.mean(1)


def criterion(y_true: torch.Tensor, y_pred: torch.Tensor):
    # restore image to their original range
    y_true, y_pred = data.reverse(y_true), data.reverse(y_pred)
    score = ssim(x=y_true, y=y_pred)
    return 1 - score


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
        loss = criterion(y_true=images, y_pred=outputs)
        reg_loss = model.regularizer()
        total_loss = loss + args.reg_scale * reg_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
            loss = criterion(y_true=images, y_pred=outputs)
            utils.update_dict(
                results,
                {"loss/loss": loss.item()},
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
            del loss, outputs
    for k, v in results.items():
        results[k] = np.mean(v)
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
    return results
