from .core import register, Core

import torch
import numpy as np
from torch import nn


@register("linear")
class LinearCore(Core):
    def __init__(self, args, input_shape: tuple, name: str = "LinearCore"):
        super(LinearCore, self).__init__(args, input_shape=input_shape, name=name)
        self.output_shape = input_shape
        self.reg_scale = torch.tensor(args.core_reg_scale, device=args.device)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=int(np.prod(input_shape)),
            out_features=int(np.prod(self.output_shape)),
        )

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.shape[0]
        outputs = self.flatten(inputs)
        outputs = self.linear(outputs)
        return outputs.view((batch_size,) + self.shape)
