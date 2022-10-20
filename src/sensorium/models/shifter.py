import torch
import typing as t
from torch import nn
from torch.nn import ModuleDict


class MLP(nn.Module):
    def __init__(
        self,
        args,
        in_features: int = 2,
        hidden_features: int = 10,
        num_layers: int = 1,
        name: str = "MLPShifter",
    ):
        """
        Multi-layer perceptron shifter
        Args:
            input_features (int): number of input features, defaults to 2.
            hidden_channels (int): number of hidden units.
            shift_layers(int): number of shifter layers (n=1 will correspond
                to a network without a hidden layer).
            **kwargs:
        """
        super(MLP, self).__init__()
        self.name = name
        self.device = args.device
        self.reg_scale = torch.tensor(args.shifter_reg_scale, device=self.device)
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

    def forward(self, pupil_center: torch.Tensor, trial_idx: torch.Tensor = None):
        if trial_idx is not None:
            pupil_center = torch.cat((pupil_center, trial_idx), dim=1)
        return self.mlp(pupil_center)


class MLPShifter(ModuleDict):
    def __init__(
        self,
        args,
        mouse_ids: t.List[int],
        input_channels: int = 2,
        hidden_features: int = 5,
        num_layers: int = 3,
    ):
        """
        Args:
            data_keys (list of str): keys of the shifter dictionary,
                correspond to the data_keys of the nnfabirk dataloaders
            gamma_shifter: weight of the regularizer
            See docstring of base class for the other arguments.
        """
        super().__init__()
        for mouse_id in mouse_ids:
            self.add_module(
                name=str(mouse_id),
                module=MLP(
                    args,
                    in_features=input_channels,
                    hidden_features=hidden_features,
                    num_layers=num_layers,
                    name=f"Mouse{mouse_id}Shifter",
                ),
            )

    def regularizer(self, mouse_id: int):
        return self[str(mouse_id)].regularizer()

    def forward(
        self,
        mouse_id: int,
        pupil_center: torch.Tensor,
        trial_idx: torch.Tensor = None,
    ):
        return self[str(mouse_id)](pupil_center, trial_idx)
