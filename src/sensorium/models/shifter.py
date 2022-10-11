import torch
import typing as t
from torch import nn
from torch.nn import ModuleDict
from torch.nn.init import xavier_normal


class Shifter(nn.Module):
    """
    Abstract base class for a shifter. It's strongly advised that the
    regularizer and initialize methods are implemented by the inheriting class.
    """

    def __init__(self, name: str = "Shifter"):
        super(Shifter, self).__init__()
        self.name = name

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"

    def regularizer(self):
        """
        Regularizer method to be used during training.
        """
        raise NotImplementedError(
            "regularizer method must be implemented by the inheriting class"
        )

    def initialize(self):
        """
        weight initialization of the torch.parameters
        """
        raise NotImplementedError(
            "initialize method must be implemented by the inheriting class"
        )


class MLP(Shifter):
    def __init__(
        self,
        input_features: int = 2,
        hidden_channels: int = 10,
        shift_layers: int = 1,
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
        super(MLP, self).__init__(name=name)

        prev_output = input_features
        feat = []
        for _ in range(shift_layers - 1):
            feat.extend([nn.Linear(prev_output, hidden_channels), nn.Tanh()])
            prev_output = hidden_channels

        feat.extend([nn.Linear(prev_output, 2), nn.Tanh()])
        self.mlp = nn.Sequential(*feat)

    def regularizer(self):
        return 0

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, pupil_center: torch.Tensor, trial_idx: torch.Tensor = None):
        if trial_idx is not None:
            pupil_center = torch.cat((pupil_center, trial_idx), dim=1)
        return self.mlp(pupil_center)


class MLPShifter(ModuleDict):
    def __init__(
        self,
        mouse_ids: t.List[int],
        input_channels: int = 2,
        hidden_channels_shifter: int = 5,
        shift_layers: int = 1,
        gamma_shifter: float = 0,
    ):
        """
        Args:
            data_keys (list of str): keys of the shifter dictionary,
                correspond to the data_keys of the nnfabirk dataloaders
            gamma_shifter: weight of the regularizer
            See docstring of base class for the other arguments.
        """
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for mouse_id in mouse_ids:
            self.add_module(
                name=str(mouse_id),
                module=MLP(
                    input_features=input_channels,
                    hidden_channels=hidden_channels_shifter,
                    shift_layers=shift_layers,
                    name=f"Mouse{mouse_id}Shifter",
                ),
            )

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def regularizer(self, mouse_id: int):
        return self[str(mouse_id)].regularizer() * self.gamma_shifter

    def forward(
        self,
        mouse_id: int,
        pupil_center: torch.Tensor,
        trial_idx: torch.Tensor = None,
    ):
        return self[str(mouse_id)](pupil_center, trial_idx)
