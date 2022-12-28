import torch
import numpy as np
import typing as t
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from sensorium.models.utils import BufferDict

_CRITERION = dict()


def register(name):
    def add_to_dict(fn):
        global _CRITERION
        _CRITERION[name] = fn
        return fn

    return add_to_dict


REDUCTION = t.Literal["sum", "mean"]
EPS = torch.finfo(torch.float32).smallest_normal


def msse(y_true: torch.Tensor, y_pred: torch.Tensor, reduction: REDUCTION = "sum"):
    """Mean sum squared error"""
    loss = torch.square(y_true - y_pred)
    loss = torch.sum(loss, dim=-1)  # sum over neurons
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def poisson_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: t.Union[float, torch.Tensor] = 1e-12,
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
    eps: t.Union[torch.Tensor, float] = 1e-8,
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
        reduction: REDUCTION = "sum",
    ):
        super(Loss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )
        self.ds_scale = args.ds_scale
        self.ds_sizes = BufferDict(
            buffers={
                str(mouse_id): torch.tensor(len(mouse_ds.dataset), dtype=torch.float32)
                for mouse_id, mouse_ds in ds.items()
            }
        )

    def scale_ds(self, loss: torch.Tensor, mouse_id: int, batch_size: int):
        """Scale loss based on the size of the dataset"""
        if self.ds_scale:
            scale = torch.sqrt(self.ds_sizes[str(mouse_id)] / batch_size)
            loss = scale * loss
        return loss


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
    def __init__(
        self,
        args,
        ds: t.Dict[int, DataLoader],
        eps: float = EPS,
        reduction: REDUCTION = "sum",
    ):
        super(PoissonLoss, self).__init__(args, ds=ds, reduction=reduction)
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, mouse_id: int):
        print(f"y_true min: {torch.min(y_true):.04f} max: {torch.max(y_true):.04f}")
        print(f"y_pred min: {torch.min(y_pred):.04f} max: {torch.max(y_pred):.04f}")
        loss = poisson_loss(y_true, y_pred, eps=self.eps, reduction=self.reduction)
        print(f"loss before scale: {loss:.02f}")
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        print(f"loss after scale: {loss:.02f}\n\n")
        return loss


@register("correlation")
class Correlation(Loss):
    """single trial correlation"""

    def __init__(self, args, ds: t.Dict[int, DataLoader], eps: float = EPS):
        super(Correlation, self).__init__(args, ds=ds)
        self.register_buffer("eps", torch.tensor(eps))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: int,
    ):
        num_neurons = y_true.size(1)
        corr = correlation(y1=y_true, y2=y_pred, dim=0, eps=self.eps)
        loss = num_neurons - torch.sum(corr)
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=y_true.size(0))
        return loss


def get_criterion(args, ds: t.Dict[int, DataLoader]):
    assert args.criterion in _CRITERION, f"Criterion {args.criterion} not found."
    criterion = _CRITERION[args.criterion](args, ds=ds)
    criterion.to(args.device)
    return criterion
