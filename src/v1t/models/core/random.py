from .core import register, Core

import torch
import numpy as np
from torch import nn


@register("random")
class RandomCore(Core):
    def __init__(self, args, input_shape: tuple, name: str = "RandomCore"):
        super(RandomCore, self).__init__(args, input_shape=input_shape, name=name)
        self.output_shape = input_shape

        self.weight = nn.Parameter(torch.rand(1))

    def regularizer(self):
        return 0.0

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.size(0)
        random = torch.rand(*(batch_size, *self.output_shape), device=inputs.device)
        return random + self.weight - self.weight
