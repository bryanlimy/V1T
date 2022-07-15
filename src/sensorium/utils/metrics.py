import torch
import typing as t


def correlation(
    y1: torch.Tensor,
    y2: torch.Tensor,
    dim: t.Union[None, int, t.Tuple[int]],
    eps: float = 1e-8,
):
    """Compute the correlation between y1 and y2 along dimensions dim."""
    y1 = (y1 - torch.mean(y1, dim=dim, keepdim=True)) / (
        torch.std(y1, dim=dim, keepdim=True) + eps
    )
    y2 = (y2 - torch.mean(y2, dim=dim, keepdim=True)) / (
        torch.std(y2, dim=dim, keepdim=True) + eps
    )
    return torch.mean(y1 * y2, dim=dim)
