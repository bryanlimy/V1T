import torch
import typing as t
import numpy as np
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

_CRITERION = dict()


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
        self._ds_scale = args.ds_scale
        self._get_ds_sizes(ds)

    def _get_ds_sizes(self, ds: t.Dict[int, DataLoader]):
        self._ds_sizes = {
            mouse_id: torch.tensor(
                len(mouse_ds.dataset), dtype=torch.int32, device=self._device
            )
            for mouse_id, mouse_ds in ds.items()
        }

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
        loss = torch.sqrt(torch.mean(torch.sum(loss, dim=-1)))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        return loss


@register("poisson")
class PoissonLoss(Loss):
    def __init__(self, args, ds: t.Dict[int, DataLoader], eps: float = 1e-8):
        super(PoissonLoss, self).__init__(args, ds=ds)
        self.eps = eps

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, mouse_id: int):
        loss = y_pred - y_true * torch.log(y_pred + self.eps)
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
    ):
        num_neurons = y_true.size(1)
        dim = 0  # compute correlation over batch dimension
        y1 = (y_true - y_true.mean(dim=dim)) / (
            y_true.std(dim=dim, unbiased=False) + eps
        )
        y2 = (y_pred - y_pred.mean(dim=dim)) / (
            y_pred.std(dim=dim, unbiased=False) + eps
        )
        corr = (y1 * y2).mean(dim=dim)
        loss = num_neurons - torch.sum(corr)
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        return loss


def get_criterion(args, ds: t.Dict[int, DataLoader]):
    assert args.criterion in _CRITERION, f"Criterion {args.criterion} not found."
    return _CRITERION[args.criterion](args, ds=ds)
