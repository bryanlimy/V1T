_CORES = dict()

import torch
import typing as t
from torch import nn


def register(name):
    """Note: update __init__.py so that register works"""

    def add_to_dict(fn):
        global _CORES
        _CORES[name] = fn
        return fn

    return add_to_dict


class Core(nn.Module):
    def __init__(self, args, input_shape: t.Tuple[int, int, int], name: str = "Conv"):
        super(Core, self).__init__()
        self.input_shape = input_shape
        self.name = name
        self._device = args.device
        self.reg_scale = torch.tensor(
            args.reg_scale, dtype=torch.float32, device=self._device
        )

    @property
    def shape(self):
        return self._output_shape

    def initialize(self):
        pass

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())


def get_core(args):
    if not args.core in _CORES.keys():
        raise NotImplementedError(f"Core {args.core} has not been implemented.")
    return _CORES[args.core]
