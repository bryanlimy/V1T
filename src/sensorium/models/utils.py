import torch
import numpy as np
import typing as t
from torch import nn
from math import floor


def int2tuple(value: t.Union[int, t.Tuple[int, int]]):
    return (value, value) if isinstance(value, int) else value


def conv2d_shape(
    input_shape: t.Tuple[int, int, int],
    num_filters: int,
    kernel_size: t.Union[int, t.Tuple[int, int]],
    stride: t.Union[int, t.Tuple[int, int]] = 1,
    padding: t.Union[int, t.Tuple[int, int]] = 0,
    dilation: t.Union[int, t.Tuple[int, int]] = 1,
):
    """Calculate 2D convolution output shape given input shape in shape (C,H,W)"""
    kernel_size = int2tuple(kernel_size)
    stride = int2tuple(stride)
    padding = int2tuple(padding)
    dilation = int2tuple(dilation)
    new_h = (
        (input_shape[1] + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1)
        / stride[0]
    ) + 1

    new_w = (
        (input_shape[2] + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1)
        / stride[1]
    ) + 1

    return (num_filters, floor(new_h), floor(new_w))


def pool2d_shape(
    input_shape: t.Tuple[int, int, int],
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
):
    if stride is None:
        stride = kernel_size

    new_h = (
        (input_shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
    ) + 1
    new_w = (
        (input_shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
    ) + 1
    return (input_shape[0], floor(new_h), floor(new_w))


def transpose_conv2d_shape(
    input_shape: t.Tuple[int, int, int],
    num_filters: int,
    kernel_size: t.Union[int, t.Tuple[int, int]],
    stride: t.Union[int, t.Tuple[int, int]] = 1,
    padding: t.Union[int, t.Tuple[int, int]] = 0,
    output_padding: t.Union[int, t.Tuple[int, int]] = 0,
    dilation: t.Union[int, t.Tuple[int, int]] = 1,
):
    kernel_size = int2tuple(kernel_size)
    stride = int2tuple(stride)
    padding = int2tuple(padding)
    output_padding = int2tuple(output_padding)
    dilation = int2tuple(dilation)
    new_h = (
        (input_shape[1] - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    new_w = (
        (input_shape[2] - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )
    return (num_filters, floor(new_h), floor(new_w))


class DropPath(nn.Module):
    """
    Stochastic depth for regularization https://arxiv.org/abs/1603.09382
    Reference:
    - https://github.com/aanna0701/SPT_LSA_ViT/blob/main/utils/drop_path.py
    - https://github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dropout: float = 0.0):
        super(DropPath, self).__init__()
        assert 0 <= dropout <= 1
        self.register_buffer("keep_prop", torch.tensor(1 - dropout))

    def forward(self, inputs: torch.Tensor):
        if self.keep_prop == 1 or not self.training:
            return inputs
        shape = (inputs.size(0),) + (1,) * (inputs.ndim - 1)
        random_tensor = torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        random_tensor = torch.floor(self.keep_prop + random_tensor)
        outputs = (inputs / self.keep_prop) * random_tensor
        return outputs


def init_weights(module: nn.Module):
    """Weight initialization for module"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)
