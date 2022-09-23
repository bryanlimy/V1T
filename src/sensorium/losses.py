import torch
from torch import nn


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.mean(torch.square(y_true - y_pred))


def mean_sum_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.mean(torch.sum(torch.square(y_true - y_pred), dim=-1))
