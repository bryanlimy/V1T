import torch
import numpy as np
from torch import nn
from math import floor


def conv2d_output_shape(
    input_shape: tuple,
    num_filters: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    """Calculate Conv2d output shape given input shape in shape (C,H,W)"""
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    new_h = floor(
        (
            (input_shape[1] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1)
            / stride
        )
        + 1
    )
    new_w = floor(
        (
            (input_shape[2] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1)
            / stride
        )
        + 1
    )
    return (num_filters, new_h, new_w)
