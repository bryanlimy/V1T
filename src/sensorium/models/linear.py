from .registry import register

import torch
import numpy as np
from torch import nn


@register("linear")
def get_model(args):
    return LinearModel(args)


class LinearModel(nn.Module):
    def __init__(self, args):
        super(LinearModel, self).__init__()

        self.device = args.device
        self.input_shape = args.input_shape
        self.output_shape = args.output_shape
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=int(np.prod(self.input_shape)),
                out_features=int(np.prod(self.output_shape)),
            ),
            nn.ELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs) + 1
