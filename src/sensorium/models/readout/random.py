from .readout import register, Readout

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader


@register("random")
class RandomReadout(Readout):
    def __init__(
        self,
        args,
        input_shape: tuple,
        output_shape: tuple,
        ds: DataLoader,
        name: str = "RandomReadout",
    ):
        super(RandomReadout, self).__init__(
            args,
            input_shape=input_shape,
            output_shape=output_shape,
            ds=ds,
            name=name,
        )

        self.weight = nn.Parameter(torch.rand(1))

    def forward(self, inputs: torch.Tensor, shift: torch.Tensor = None):
        batch_size = inputs.size(0)
        return torch.rand(*(batch_size, *self.output_shape)) + self.weight - self.weight
