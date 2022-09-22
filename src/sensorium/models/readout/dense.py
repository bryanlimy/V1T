from .readout import register

import torch
import numpy as np
from torch import nn


@register("dense")
class DenseReadout(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, name: str = None):
        super(DenseReadout, self).__init__()
        self.name = "LinearReadout" if name is None else name
        self._output_shape = output_shape

        out_features = int(np.prod(output_shape))
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=int(np.prod(input_shape)),
                out_features=out_features // 2,
            ),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=out_features // 2, out_features=out_features),
            nn.ELU(),
        )

    @property
    def shape(self):
        return self._output_shape

    def forward(self, inputs: torch.Tensor):
        outputs = self.dense(inputs)
        return outputs + 1
