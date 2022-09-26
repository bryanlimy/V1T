from .readout import register, Readout

import torch
import numpy as np
import typing as t
from torch import nn
from torch.nn import functional as F

REDUCTIONS = t.Literal["sum", "mean", None]


@register("gaussian2d")
class Gaussian2DReadout(Readout):
    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        use_bias: bool = True,
        init_mu_range: float = 0.1,
        init_sigma: float = 1.0,
        gaussian_type: str = "full",
        grid_mean_predictor: t.Dict[str, t.Union[float, bool]] = None,
        source_grid: np.ndarray = None,
        mean_response: np.ndarray = None,
        feature_reg_weight: float = 1.0,
        name: str = "Gaussian2DReadout",
    ):
        super(Gaussian2DReadout, self).__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            mean_response=mean_response,
            name=name,
        )

        self.feature_reg_weight = feature_reg_weight

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError(
                "either init_mu_range doesn't belong to [0.0, 1.0] or "
                "init_sigma_range is non-positive"
            )
        self.init_mu_range = init_mu_range

        # position grid shape
        self.grid_shape = (1, self.shape[-1], 1, 2)

        # the grid can be predicted from another grid
        self._predicted_grid = False
        self._original_grid = not self._predicted_grid

        if grid_mean_predictor is None:
            # mean location of gaussian for each neuron
            self._mu = nn.Parameter(torch.Tensor(*self.grid_shape))
        else:
            self.init_grid_predictor(source_grid=source_grid, **grid_mean_predictor)

        self.gaussian_type = gaussian_type
        if gaussian_type == "full":
            self.sigma_shape = (1, self.shape[-1], 2, 2)
        elif gaussian_type == "uncorrelated":
            self.sigma_shape = (1, self.shape[-1], 1, 2)
        elif gaussian_type == "isotropic":
            self.sigma_shape = (1, self.shape[-1], 1, 1)
        else:
            raise ValueError(f"Unknown Gaussian type {gaussian_type}.")

        self.init_sigma = init_sigma
        # standard deviation for gaussian for each neuron
        self.sigma = nn.Parameter(torch.Tensor(*self.sigma_shape))

        self.initialize_features()

        bias = None
        if use_bias:
            bias = nn.Parameter(torch.Tensor(self.shape[-1]))
        self.register_parameter("bias", bias)

        self.mean_response = mean_response
        self.initialize(mean_response=mean_response)

    def feature_l1(self, reduction: REDUCTIONS = "sum"):
        """
        Returns l1 regularization term for features.
        Args:
            reduction(str): Specifies the reduction to apply to the output:
                            'none' | 'mean' | 'sum'
        """
        l1 = 0
        if self._original_features:
            l1 = self.features.abs()
            if reduction == "sum":
                l1 = l1.sum()
            elif reduction == "mean":
                l1 = l1.mean()
        return l1

    def regularizer(self, reduction: REDUCTIONS = "sum"):
        return self.feature_reg_weight * self.feature_l1(reduction=reduction)

    def init_grid_predictor(
        self,
        source_grid: np.ndarray,
        hidden_features: int = 20,
        hidden_layers: int = 0,
        tanh_output: bool = False,
    ):
        self._original_grid = False
        layers = [
            nn.Linear(
                in_features=source_grid.shape[1],
                out_features=hidden_features if hidden_layers > 0 else 2,
            )
        ]

        for i in range(hidden_layers):
            layers.extend(
                [
                    nn.ELU(),
                    nn.Linear(
                        in_features=hidden_features,
                        out_features=hidden_features if i < hidden_layers - 1 else 2,
                    ),
                ]
            )

        if tanh_output:
            layers.append(nn.Tanh())

        self.mu_transform = nn.Sequential(*layers)

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid))
        self._predicted_grid = True

    def initialize_features(self):
        """
        The internal attribute `_original_features` in this function denotes
        whether this instance of the FullGaussian2d learns the original
        features (True) or if it uses a copy of the features from another
        instance of FullGaussian2d via the `shared_features` (False). If it
        uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self._input_shape
        self._original_features = True

        # feature weights for each channel of the core
        self._features = nn.Parameter(torch.Tensor(1, c, 1, self.outdims))
        self._shared_features = False

    def initialize(self, mean_response: np.ndarray = None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with
        the features weights
        """
        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gaussian_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        self._features.data.fill_(1 / self.in_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.0)

        if mean_response is None:
            mean_response = self.mean_responses
        if self.bias is not None:
            self.initialize_bias(mean_response=mean_response)

    def initialize_bias(self, mean_response: np.ndarray = None) -> None:
        """Initialize the biases in readout"""
        if mean_response is None:
            self.bias.data.fill_(0)
        else:
            self.bias.data = torch.from_numpy(mean_response)

    def forward(self, inputs: torch.Tensor, sample: bool = None, shift: bool = None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample
                                from Gaussian distribution, N(mu,sigma),
                                defined per neuron or use the mean, mu, of the
                                Gaussian distribution without sampling.
                                If sample is None (default), samples from the
                                N(mu,sigma) during training phase and fixes to
                                the mean, mu, during evaluation phase.
                                If sample is True/False, overrides the
                                model_state (i.e. training or eval) and does
                                as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        batch_size, c, w, h = inputs.size()
        c_in, w_in, h_in = self._input_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError(
                f"shape mismatch between expected ({self._input_shape}) and "
                f"received ({inputs.size()}) inputs."
            )
        features = self.features.view(1, c, self.outdims)
        bias = self.bias
        num_neurons = self.shape[-1]

        # sample the grid_locations separately per image per batch
        grid = self.sample_grid(batch_size=batch_size, sample=sample)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        outputs = F.grid_sample(inputs, grid=grid, align_corners=True)
        outputs = (outputs.squeeze(-1) * features).sum(1).view(batch_size, num_neurons)

        if self.bias is not None:
            outputs = outputs + bias
        return outputs


def prepare_grid(grid_mean_predictor, dataloaders):
    """
    Utility function for using the neurons cortical coordinates
    to guide the readout locations in image space.

    Args:
        grid_mean_predictor (dict): config dictionary, for example:
          {'type': 'cortex',
           'input_dimensions': 2,
           'hidden_layers': 1,
           'hidden_features': 30,
           'final_tanh': True}

        dataloaders: a dictionary of dataloaders, one PyTorch DataLoader per session
            in the format {'data_key': dataloader object, .. }
    Returns:
        grid_mean_predictor (dict): config dictionary
        grid_mean_predictor_type (str): type of the information that is being used for
            the grid position estimator
        source_grids (dict): a grid of points for each data_key

    """
    if grid_mean_predictor is None:
        grid_mean_predictor_type = None
        source_grids = None
    else:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")

        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {
                k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim]
                for k, v in dataloaders.items()
            }
    return grid_mean_predictor, grid_mean_predictor_type, source_grids
