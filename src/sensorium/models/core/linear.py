from .core import register

import torch
import numpy as np
from torch import nn


@register("linear")
class LinearCore(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, name: str = None):
        super(LinearCore, self).__init__()
        self.name = "LinearReadout" if name is None else name
        self._output_shape = output_shape

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=int(np.prod(input_shape)),
            out_features=int(np.prod(output_shape)),
        )

    @property
    def shape(self):
        return self._output_shape

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.shape[0]
        outputs = self.flatten(inputs)
        outputs = self.linear(outputs)
        return outputs.view((batch_size,) + self.shape)
