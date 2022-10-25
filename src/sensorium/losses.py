import torch
import typing as t
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss


REDUCTION = t.Literal["sum", "mean"]

_CRITERION = dict()


def register(name):
    def add_to_dict(fn):
        global _CRITERION
        _CRITERION[name] = fn
        return fn

    return add_to_dict


def msse(y_true: torch.Tensor, y_pred: torch.Tensor, reduction: REDUCTION = "sum"):
    """Mean sum squared error"""
    loss = torch.square(y_true - y_pred)
    loss = torch.sum(loss, dim=-1)  # sum over neurons
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def poisson_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-12,
    reduction: REDUCTION = "sum",
):
    loss = y_pred - y_true * torch.log(y_pred + eps)
    loss = torch.sum(loss, dim=-1)  # sum over neurons
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def _t_correlation(
    y1: torch.Tensor,
    y2: torch.Tensor,
    dim: t.Union[None, int, t.Tuple[int]] = -1,
    eps: float = 1e-8,
):
    with autocast(enabled=False):
        if dim is None:
            dim = tuple(range(y1.dim()))
        y1 = (y1 - y1.mean(dim=dim, keepdim=True)) / (
            y1.std(dim=dim, unbiased=False, keepdim=True) + eps
        )
        y2 = (y2 - y2.mean(dim=dim, keepdim=True)) / (
            y2.std(dim=dim, unbiased=False, keepdim=True) + eps
        )
        corr = (y1 * y2).mean(dim=dim)
    return corr


def _np_correlation(
    y1: np.ndarray,
    y2: np.ndarray,
    axis: t.Union[None, int, t.Tuple[int]] = -1,
    eps: float = 1e-8,
    **kwargs,
):
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (
        y1.std(axis=axis, ddof=0, keepdims=True) + eps
    )
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (
        y2.std(axis=axis, ddof=0, keepdims=True) + eps
    )
    corr = (y1 * y2).mean(axis=axis, **kwargs)
    return corr


def correlation(
    y1: t.Union[torch.Tensor, np.ndarray],
    y2: t.Union[torch.Tensor, np.ndarray],
    dim: t.Union[None, int, t.Tuple[int]] = -1,
    eps: float = 1e-8,
    **kwargs,
):
    return (
        _t_correlation(y1=y1, y2=y2, dim=dim, eps=eps)
        if isinstance(y1, torch.Tensor)
        else _np_correlation(y1=y1, y2=y2, axis=dim, eps=eps, **kwargs)
    )


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


@register("msse")
class MSSE(Loss):
    """mean sum squared error"""

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: int,
        reduction: REDUCTION = "sum",
    ):
        loss = msse(y_true=y_true, y_pred=y_pred, reduction=reduction)
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        return loss


@register("poisson")
class PoissonLoss(Loss):
    def __init__(self, args, ds: t.Dict[int, DataLoader]):
        super(PoissonLoss, self).__init__(args, ds=ds)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: int,
        reduction: REDUCTION = "sum",
        eps: float = 1e-12,
    ):
        print(f"\n{y_true.dtype}, {y_pred.dtype}, {eps}")
        exit()
        loss = poisson_loss(y_true, y_pred, eps=eps, reduction=reduction)
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
        corr = correlation(y1=y_true, y2=y_pred, dim=0, eps=eps)
        loss = num_neurons - torch.sum(corr)
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        return loss


def get_criterion(args, ds: t.Dict[int, DataLoader]):
    assert args.criterion in _CRITERION, f"Criterion {args.criterion} not found."
    return _CRITERION[args.criterion](args, ds=ds)
