import torch
import typing as t
from torch import nn
from torch.nn import ModuleDict


class CoreShifter(nn.Module):
    """
    Code reference: https://github.com/sinzlab/neuralpredictors/blob/9b85300ab854be1108b4bf64b0e4fa2e960760e0/neuralpredictors/layers/shifters/mlp.py
    """

    def __init__(
        self,
        args: t.Any,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        name: str = "CoreShifter",
    ):
        super(CoreShifter, self).__init__()
        self.name = name
        self.register_buffer("reg_scale", torch.tensor(args.shifter_reg_scale))
        out_features = in_features
        layers = []
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_features=out_features, out_features=hidden_features),
                    nn.Tanh(),
                ]
            )
            out_features = hidden_features
        layers.extend([nn.Linear(in_features=out_features, out_features=2), nn.Tanh()])
        self.mlp = nn.Sequential(*layers)

    def regularizer(self):
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, pupil_center: torch.Tensor):
        return self.mlp(pupil_center)


class CoreShifters(ModuleDict):
    def __init__(
        self,
        args: t.Any,
        mouse_ids: t.List[str],
        input_channels: int,
        hidden_features: int,
        num_layers: int,
    ):
        super(CoreShifters, self).__init__()
        for mouse_id in mouse_ids:
            self.add_module(
                name=mouse_id,
                module=CoreShifter(
                    args,
                    in_features=input_channels,
                    hidden_features=hidden_features,
                    num_layers=num_layers,
                    name=f"Mouse{mouse_id}CoreShifter",
                ),
            )

    def regularizer(self, mouse_id: str):
        return self[mouse_id].regularizer()

    def forward(self, pupil_centers: torch.Tensor, mouse_id: str):
        return self[mouse_id](pupil_centers)
