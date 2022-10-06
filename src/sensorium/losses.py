import torch
import typing as t
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

_CRITERION = dict()
import numpy as np


def register(name):
    def add_to_dict(fn):
        global _CRITERION
        _CRITERION[name] = fn
        return fn

    return add_to_dict


class Loss(_Loss):
    """Basic Criterion class"""

    def __init__(
        self,
        args,
        ds: t.Dict[int, DataLoader],
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
    ):
        super(Loss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )
        self._device = args.device
        self._depth_scale = args.depth_scale
        self._ds_scale = args.ds_scale
        self._compute_depth_masks(ds)
        self._get_ds_sizes(ds)

    def _compute_depth_masks(self, ds: t.Dict[int, DataLoader]):
        """Extract the weighted loss mask based on the depth (z-axis) of neurons

        Neurons between the depth of 240 to 260 are masked by self._depth_scale,
        otherwise 1.
        """
        self._depth_masks = {}
        for mouse_id, mouse_ds in ds.items():
            depth = torch.from_numpy(mouse_ds.dataset.coordinates[:, -1])
            mask = torch.where((depth >= 240) & (depth <= 260), self._depth_scale, 1)
            self._depth_masks[mouse_id] = mask.to(self._device)

    def _get_ds_sizes(self, ds: t.Dict[int, DataLoader]):
        self._ds_sizes = {
            mouse_id: torch.tensor(
                len(mouse_ds.dataset), dtype=torch.int32, device=self._device
            )
            for mouse_id, mouse_ds in ds.items()
        }

    def scale_neurons(self, loss: torch.Tensor, mouse_id: int):
        return loss * self._depth_masks[mouse_id]

    def scale_ds(self, loss: torch.Tensor, mouse_id: int, batch_size: int):
        """Scale loss based on the size of the dataset"""
        loss_scale = (
            torch.sqrt(self._ds_sizes[mouse_id] / batch_size) if self._ds_scale else 1.0
        )
        return loss_scale * loss


@register("rmsse")
class RMSSE(Loss):
    """Root-mean-sum-squared-error"""

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, mouse_id: int):
        loss = torch.square(y_true - y_pred)
        loss = self.scale_neurons(loss, mouse_id=mouse_id)
        loss = torch.sqrt(torch.mean(torch.sum(loss, dim=-1)))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        return loss


@register("poisson")
class PoissonLoss(Loss):
    """
    Poisson Loss

    Computes Poisson loss between the output and target.
    Loss is evaluated by computing log likelihood (up to a constant offset
    dependent on the target) that output prescribes the mean of the
    Poisson distribution and target is a sample from the distribution.

    Args:
        eps (float, optional): Value used to numerically stabilize
            evaluation of the log-likelihood. This value is effectively
            added to the output during evaluation.
            Defaults to 1e-12.
        per_neuron (bool, optional): If set to True, the average/total
            Poisson loss is returned for each entry of the last dimension
            (assumed to be enumeration neurons) separately.
            Defaults to False.
        return_average (bool, optional): If set to True, return mean loss.
            Otherwise, returns the sum of loss.
            Defaults to False.
    """

    def __init__(self, args, ds: t.Dict[int, DataLoader], eps: float = 1e-8):
        super(PoissonLoss, self).__init__(args, ds=ds)
        self.eps = eps

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, mouse_id: int):
        loss = y_pred - y_true * torch.log(y_pred + self.eps)
        loss = self.scale_neurons(loss, mouse_id=mouse_id)
        loss = torch.mean(torch.sum(loss, dim=-1))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        return loss


@register("correlation")
class Correlation(Loss):
    """single trial correlation"""

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: int,
        eps: float = 1e-8,
        dim: int = 0,
    ):
        y1 = (y_true - y_true.mean(dim=dim, keepdim=True)) / (
            y_true.std(dim=dim, keepdim=True, unbiased=False) + eps
        )
        y2 = (y_pred - y_pred.mean(dim=dim, keepdim=True)) / (
            y_pred.std(dim=dim, keepdim=True, unbiased=False) + eps
        )
        corr = (y1 * y2).mean(dim=dim)
        loss = 1.0 - corr.mean()
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        return loss


def get_criterion(args, ds: t.Dict[int, DataLoader]):
    assert args.criterion in _CRITERION, f"Criterion {args.criterion} not found."
    return _CRITERION[args.criterion](args, ds=ds)
