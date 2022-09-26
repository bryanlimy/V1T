from .readout import register, Readout

import torch
import numpy as np
from torch import nn


@register("linear")
class LinearReadout(Readout):
    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        mean_response: np.ndarray = None,
        name: str = "LinearReadout",
    ):
        super(LinearReadout, self).__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            mean_response=mean_response,
            name=name,
        )

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
        outputs = outputs.view((batch_size,) + self.shape)
        return outputs
