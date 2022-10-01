import torch
from torch import nn
from torch.nn.modules.loss import _Loss

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
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
    ):
        super(Loss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )


@register("rmsse")
class RMSSE(Loss):
    """Root-mean-sum-squared-error"""

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return torch.sqrt(torch.mean(torch.sum(torch.square(y_true - y_pred), dim=-1)))


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

    def __init__(
        self,
        eps: float = 1e-12,
        per_neuron: bool = False,
        return_average: bool = False,
    ):
        super(PoissonLoss, self).__init__()
        self.eps = eps
        self._per_neuron = per_neuron
        self._return_average = return_average

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.detach()
        loss = y_pred - y_true * torch.log(y_pred + self.eps)
        if self._per_neuron:
            loss = loss.view(-1, loss.shape[-1])
            loss = loss.mean(dim=0) if self._return_average else loss.sum(dim=0)
        else:
            loss = loss.mean() if self._return_average else loss.sum()
        return loss


def get_criterion(args):
    assert args.criterion in _CRITERION, f"Criterion {args.criterion} not found."
    return _CRITERION[args.criterion]()
