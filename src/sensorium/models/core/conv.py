from .core import register

import torch
import numpy as np
from torch import nn

from sensorium.models import utils


@register("conv")
class ConvCore(nn.Module):
    def __init__(
        self,
        args,
        input_shape: tuple,
        kernel_size: int = 3,
        stride: int = 2,
        name: str = None,
    ):
        super(ConvCore, self).__init__()
        self.name = "ConvCore" if name is None else name

        output_shape = input_shape
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=args.num_filters,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.InstanceNorm2d(num_features=args.num_filters),
            nn.GELU(),
            nn.Dropout2d(p=args.dropout),
        )
        output_shape = utils.conv2d_output_shape(
            output_shape,
            num_filters=args.num_filters,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=output_shape[0],
                out_channels=args.num_filters * 2,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.InstanceNorm2d(num_features=args.num_filters),
            nn.GELU(),
            nn.Dropout2d(p=args.dropout),
        )
        output_shape = utils.conv2d_output_shape(
            output_shape,
            num_filters=args.num_filters * 2,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=output_shape[0],
                out_channels=args.num_filters * 3,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.InstanceNorm2d(num_features=args.num_filters),
            nn.GELU(),
            nn.Dropout2d(p=args.dropout),
        )
        output_shape = utils.conv2d_output_shape(
            output_shape,
            num_filters=args.num_filters * 3,
            kernel_size=kernel_size,
            stride=stride,
        )
        self._output_shape = output_shape

    @property
    def shape(self):
        return self._output_shape

    def forward(self, inputs: torch.Tensor):
        outputs = self.conv_block1(inputs)
        outputs = self.conv_block2(outputs)
        outputs = self.conv_block3(outputs)
        return outputs
