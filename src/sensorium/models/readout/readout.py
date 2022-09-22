_READOUTS = dict()

import torch
import typing as t
from torch import nn


def register(name):
    """Note: update __init__.py so that register works"""

    def add_to_dict(fn):
        global _READOUTS
        _READOUTS[name] = fn
        return fn

    return add_to_dict


class Readouts(nn.ModuleDict):
    def __init__(
        self,
        model: str,
        input_shape: t.Tuple[int],
        output_shapes: t.Dict[str, tuple],
    ):
        super(Readouts, self).__init__()
        if model not in _READOUTS.keys():
            raise NotImplementedError(f"Readout {model} has not been implemented.")
        self.input_shape = input_shape
        self.output_shapes = output_shapes
        readout_model = _READOUTS[model]
        for mouse_id, output_shape in self.output_shapes.items():
            self.add_module(
                name=str(mouse_id),
                module=readout_model(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    name=f"Mouse{mouse_id}Readout",
                ),
            )

    def forward(self, inputs: torch.Tensor, mouse_id: torch.Union[int, torch.Tensor]):
        return self[str(mouse_id)](inputs)
