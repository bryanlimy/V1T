import logging

import torch
import warnings
from torch import nn
from torch.nn import ModuleDict
from torch.nn.init import xavier_normal

logger = logging.getLogger(__name__)


class Shifter(nn.Module):
    """
    Abstract base class for a shifter. It's strongly adviced that the regularizer and initialize methods are implemented by the inheriting class.
    """

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
    def __init__(self, input_features=2, hidden_channels=10, shift_layers=1, **kwargs):
        """
        Multi-layer perceptron shifter
        Args:
            input_features (int): number of input features, defaults to 2.
            hidden_channels (int): number of hidden units.
            shift_layers(int): number of shifter layers (n=1 will correspond to a network without a hidden layer).
            **kwargs:
        """
        super().__init__()

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

    def forward(self, pupil_center, trial_idx=None):
        if trial_idx is not None:
            pupil_center = torch.cat((pupil_center, trial_idx), dim=1)
        if not self.mlp[0].in_features == pupil_center.shape[1]:
            raise ValueError(
                "The expected input shape of the shifter and the shape of the input do not match! "
                "(Maybe due to the appending of trial_idx to pupil_center?)"
            )
        return self.mlp(pupil_center)


class MLPShifter(ModuleDict):
    def __init__(
        self,
        data_keys,
        input_channels=2,
        hidden_channels_shifter=2,
        shift_layers=1,
        gamma_shifter=0,
        **kwargs
    ):
        """
        Args:
            data_keys (list of str): keys of the shifter dictionary, correspond to the data_keys of the nnfabirk dataloaders
            gamma_shifter: weight of the regularizer
            See docstring of base class for the other arguments.
        """
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(
                k, MLP(input_channels, hidden_channels_shifter, shift_layers)
            )

    def initialize(self, **kwargs):
        logger.info(
            "Ignoring input {} when initializing {}".format(
                repr(kwargs), self.__class__.__name__
            )
        )
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter


class StaticAffine2d(nn.Linear):
    def __init__(self, input_channels=2, output_channels=2, bias=True):
        """
        A simple FC network with bias between input and output channels without a hidden layer.
        Args:
            input_channels (int): number of input channels.
            output_channels (int): number of output channels.
            bias (bool): Adds a bias parameter if True.
        """
        super().__init__(input_channels, output_channels, bias=bias)

    def forward(self, x, trial_idx=None):
        if trial_idx is not None:
            warnings.warn(
                "Trial index was passed but is not used because this shifter network does not support trial indexing."
            )
        x = super().forward(x)
        return torch.tanh(x)

    def initialize(self, bias=None):
        self.weight.data.normal_(0, 1e-6)
        if self.bias is not None:
            if bias is not None:
                logger.info("Setting bias to predefined value")
                self.bias.data = bias
            else:
                self.bias.data.normal_(0, 1e-6)

    def regularizer(self):
        return self.weight.pow(2).mean()


class StaticAffine2dShifter(ModuleDict):
    def __init__(
        self, data_keys, input_channels=2, output_channels=2, bias=True, gamma_shifter=0
    ):
        """
        Args:
            data_keys (list of str): keys of the shifter dictionary, correspond to the data_keys of the nnfabirk dataloaders
            gamma_shifter: weight of the regularizer
            See docstring of base class for the other arguments.
        """
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(
                k, StaticAffine2d(input_channels, output_channels, bias=bias)
            )

    def initialize(self, bias=None):
        for k in self:
            if bias is not None:
                self[k].initialize(bias=bias[k])
            else:
                self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter
