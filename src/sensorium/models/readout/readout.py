_READOUTS = dict()

import torch
import typing as t
import numpy as np
from torch import nn


def register(name):
    """Note: update __init__.py so that register works"""

    def add_to_dict(fn):
        global _READOUTS
        _READOUTS[name] = fn
        return fn

    return add_to_dict


class Readout(nn.Module):
    """Basic readout module for a single rodent"""

    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        mean_response: np.ndarray = None,
        name: str = None,
    ):
        super(Readout, self).__init__()
        self.name = "Readout" if name is None else name
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._response_stat = mean_response

    @property
    def shape(self):
        """Get module output shape"""
        return self._output_shape

    def initialize(self, *args: t.Any, **kwargs: t.Any):
        pass

    def regularizer(self, reduction: str):
        pass


class Readouts(nn.ModuleDict):
    """Dictionary of Readout modules to store Mouse ID: Readout module pairs"""

    def __init__(
        self,
        model: str,
        input_shape: t.Tuple[int],
        output_shapes: t.Dict[str, tuple],
        response_stats: t.Dict[str, t.Dict[str, np.ndarray]],
    ):
        super(Readouts, self).__init__()
        if model not in _READOUTS.keys():
            raise NotImplementedError(f"Readout {model} has not been implemented.")
        self._input_shape = input_shape
        self._output_shapes = output_shapes
        self._response_stats = response_stats
        readout_model = _READOUTS[model]
        for mouse_id, output_shape in self.output_shapes.items():
            self.add_module(
                name=str(mouse_id),
                module=readout_model(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    mean_response=response_stats[mouse_id]["mean"],
                    name=f"Mouse{mouse_id}Readout",
                ),
            )
        self.initialize()

    def initialize(self):
        for mouse_id, readout in self.items():
            readout.initialize(mean_response=self._response_stats[mouse_id]["mean"])

    def regularizer(self, mouse_id: str = None, reduction: str = "sum"):
        return self[mouse_id].regularizer(reduction=reduction)

    def forward(self, inputs: torch.Tensor, mouse_id: torch.Union[int, torch.Tensor]):
        return self[str(mouse_id)](inputs)
