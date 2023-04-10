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
    def __init__(
        self, args: t.Any, input_shape: t.Tuple[int, int, int], name: str = "Core"
    ):
        super(Core, self).__init__()
        self.input_shape = input_shape
        self.name = name
        self.behavior_mode = args.behavior_mode
        if args.core != "vit":
            assert self.behavior_mode != 2
        self.frozen = False
        self.verbose = args.verbose

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
        self.frozen = True
        if self.verbose:
            print("Freeze core module.")

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)
        self.forzen = False
        if self.verbose:
            print("Unfreeze core module.")

    def initialize(self):
        raise NotImplementedError("initialize function has not been implemented")

    def regularizer(self):
        raise NotImplementedError("regularizer function has not been implemented")

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        raise NotImplementedError("forward function has not been implemented")


def get_core(args):
    if not args.core in _CORES.keys():
        raise NotImplementedError(f"Core {args.core} has not been implemented.")
    return _CORES[args.core]
