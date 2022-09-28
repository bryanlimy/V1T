import torch
from torch import nn


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.mean(torch.square(y_true - y_pred))


def mean_sum_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.mean(torch.sum(torch.square(y_true - y_pred), dim=-1))


def poisson_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    bias: float = 1e-12,
    per_neuron: bool = False,
    return_average: bool = False,
):
    """
    Computes Poisson loss between the output and target.
    Loss is evaluated by computing log likelihood (up to a constant offset
    dependent on the target) that output prescribes the mean of the
    Poisson distribution and target is a sample from the distribution.

    Args:
        bias (float, optional): Value used to numerically stabilize
            evaluation of the log-likelihood. This value is effectively
            added to the output during evaluation.
            Defaults to 1e-12.
        per_neuron (bool, optional): If set to True, the average/total
            Poisson loss is returned for each entry of the last dimension
            (assumed to be enumeration neurons) separately.
            Defaults to False.
        return_average (bool, optional): If set to True, return mean loss.
            Otherwise, returns the sum of loss.
            Defaults to False."""
    y_true = y_true.detach()
    loss = y_pred - y_true * torch.log(y_pred + bias)
    if per_neuron:
        loss = loss.view(-1, loss.shape[-1])
        loss = loss.mean(dim=0) if return_average else loss.sum(dim=0)
    else:
        loss = loss.mean() if return_average else loss.sum()
    return loss
