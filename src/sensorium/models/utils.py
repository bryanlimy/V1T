import torch
import numpy as np
import typing as t
from torch import nn
from math import floor


def conv2d_shape(
    input_shape: t.Tuple[int, int, int],
    num_filters: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    """Calculate 2D convolution output shape given input shape in shape (C,H,W)"""
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    new_h = (
        (input_shape[1] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1)
        / stride
    ) + 1

    new_w = (
        (input_shape[2] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1)
        / stride
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
