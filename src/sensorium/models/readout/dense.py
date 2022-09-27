from .readout import register, Readout

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader


@register("dense")
class DenseReadout(Readout):
    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        ds: DataLoader,
        name: str = "DenseReadout",
    ):
        super(DenseReadout, self).__init__(
            input_shape=input_shape, output_shape=output_shape, ds=ds, name=name
        )

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
        )

    def forward(self, inputs: torch.Tensor):
        outputs = self.dense(inputs)
        return outputs
