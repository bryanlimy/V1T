_READOUTS = dict()

import torch
import typing as t
import numpy as np
from torch import nn
from torch.utils.data import DataLoader


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
        args,
        input_shape: tuple,
        output_shape: tuple,
        ds: DataLoader,
        name: str = None,
    ):
        super(Readout, self).__init__()
        self.name = "Readout" if name is None else name
        self._input_shape = input_shape
        self.output_shape = output_shape
        self._device = args.device
        self._response_stat = {
            k: torch.from_numpy(v) for k, v in ds.dataset.stats["response"].items()
        }
        self._coordinates = ds.dataset.coordinates
        self.reg_scale = torch.tensor(
            args.reg_scale, dtype=torch.float32, device=self._device
        )

    @property
    def num_neurons(self):
        """Number of neurons to output"""
        return self.output_shape[-1]

    def initialize(self, *args: t.Any, **kwargs: t.Any):
        pass

    def regularizer(self, reduction: str):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())


class Readouts(nn.ModuleDict):
    """Dictionary of Readout modules to store Mouse ID: Readout module pairs"""

    def __init__(
        self,
        args,
        model: str,
        input_shape: t.Tuple[int],
        output_shapes: t.Dict[int, tuple],
        ds: t.Dict[int, DataLoader],
    ):
        super(Readouts, self).__init__()
        if model not in _READOUTS.keys():
            raise NotImplementedError(f"Readout {model} has not been implemented.")
        self._input_shape = input_shape
        self._output_shapes = output_shapes
        self._device = args.device
        readout_model = _READOUTS[model]
        for mouse_id, output_shape in self._output_shapes.items():
            self.add_module(
                name=str(mouse_id),
                module=readout_model(
                    args,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    ds=ds[mouse_id],
                    name=f"Mouse{mouse_id}Readout",
                ),
            )
        self.initialize()

    def initialize(self):
        for mouse_id, readout in self.items():
            readout.initialize()

    def regularizer(self, mouse_id: int, reduction: str = "sum"):
        return self[str(mouse_id)].regularizer(reduction=reduction)

    def forward(self, inputs: torch.Tensor, mouse_id: torch.Union[int, torch.Tensor]):
        return self[str(mouse_id)](inputs)
