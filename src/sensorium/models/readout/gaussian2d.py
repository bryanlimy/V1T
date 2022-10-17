from .readout import register, Readout

import torch
import numpy as np
import typing as t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

REDUCTIONS = t.Literal["sum", "mean", None]


@register("gaussian2d")
class Gaussian2DReadout(Readout):
    def __init__(
        self,
        args,
        input_shape: tuple,
        output_shape: tuple,
        ds: DataLoader,
        use_bias: bool = True,
        init_mu_range: float = 0.3,
        init_sigma: float = 0.1,
        gaussian_type: str = "full",
        name: str = "Gaussian2DReadout",
    ):
        super(Gaussian2DReadout, self).__init__(
            args,
            input_shape=input_shape,
            output_shape=output_shape,
            ds=ds,
            name=name,
        )

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError(
                "either init_mu_range doesn't belong to [0.0, 1.0] or "
                "init_sigma_range is non-positive"
            )
        self.init_mu_range = init_mu_range

        # self.gamma_readout = torch.tensor(
        #     0.0076, dtype=torch.float32, device=self.device
        # )

        # position grid shape
        self.grid_shape = (1, self.num_neurons, 1, 2)

        # the grid can be predicted from another grid
        self._predicted_grid = False
        self._original_grid = not self._predicted_grid

        if args.disable_grid_predictor:
            # mean location of gaussian for each neuron
            self._mu = nn.Parameter(torch.Tensor(*self.grid_shape))
        else:
            self.init_grid_predictor(
                source_grid=self.neuron_coordinates,
                input_dimensions=args.grid_predictor_dim,
            )

        self.gaussian_type = gaussian_type
        if gaussian_type == "full":
            self.sigma_shape = (1, self.num_neurons, 2, 2)
        elif gaussian_type == "uncorrelated":
            self.sigma_shape = (1, self.num_neurons, 1, 2)
        elif gaussian_type == "isotropic":
            self.sigma_shape = (1, self.num_neurons, 1, 1)
        else:
            raise ValueError(f"Unknown Gaussian type {gaussian_type}.")

        self.init_sigma = init_sigma
        # standard deviation for gaussian for each neuron
        self.sigma = nn.Parameter(torch.Tensor(*self.sigma_shape))

        self.initialize_features()

        self.use_bias = use_bias
        self.bias_mode = args.bias_mode
        self.initialize_bias(stats=ds.dataset.response_stats)

        self.initialize()

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
        return self.reg_scale * self.feature_l1(reduction=reduction)

    def init_grid_predictor(
        self,
        source_grid: torch.Tensor,
        hidden_features: int = 30,
        hidden_layers: int = 1,
        tanh_output: bool = True,
        input_dimensions: int = 2,
    ):
        self._original_grid = False
        source_grid = source_grid[:, :input_dimensions]

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
        c, w, h = self.input_shape
        self._original_features = True

        # feature weights for each channel of the core
        self.features = nn.Parameter(torch.Tensor(1, c, 1, self.num_neurons))
        self._shared_features = False

    def initialize_bias(self, stats: t.Dict[str, np.ndarray]):
        if self.use_bias:
            if self.bias_mode == 0:
                bias = torch.zeros(size=(len(stats["mean"]),))
            elif self.bias_mode == 1:
                bias = torch.from_numpy(stats["mean"])
            elif self.bias_mode == 2:
                bias = stats["mean"] / stats["std"]
                bias = torch.from_numpy(bias)
            else:
                raise NotImplementedError(
                    f"Gaussian2dReadout: bias mode {self.bias_mode} has not been implemented."
                )
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def initialize(self):
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
        self.features.data.fill_(1 / self.input_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.0)

    @property
    def mu(self):
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
        else:
            return self._mu

    def sample_grid(self, batch_size: t.Union[int, torch.Tensor], sample: bool = None):
        """
        Returns the grid locations from the core by sampling from a Gaussian
        distribution
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample
                                from Gaussian distribution, N(mu,sigma),
                                defined per neuron or use the mean, mu, of the
                                Gaussian distribution without sampling.
                                If sample is None (default), samples from the
                                N(mu,sigma) during training phase and fixes to
                                the mean, mu, during evaluation phase.
                                If sample is True/False, overrides the
                                model_state (i.e. training or eval) and does as
                                instructed
        """
        with torch.no_grad():
            # at eval time, only self.mu is used, so it must belong to [-1,1]
            # sigma/variance is always a positive quantity
            self.mu.clamp_(min=-1, max=1)

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            # for consistency and CUDA capability
            norm = self.mu.new(*grid_shape).zero_()

        if self.gaussian_type != "full":
            # grid locations in feature space sampled randomly around the mean self.mu
            return torch.clamp(norm * self.sigma + self.mu, min=-1, max=1)
        else:
            # grid locations in feature space sampled randomly around the mean self.mu
            return torch.clamp(
                torch.einsum("ancd,bnid->bnic", self.sigma, norm) + self.mu,
                min=-1,
                max=1,
            )

    def forward(
        self, inputs: torch.Tensor, sample: bool = None, shift: torch.Tensor = None
    ):
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
            shift (torch.Tensor): shifts the location of the grid from
                eye-tracking data
        """
        batch_size, c, w, h = inputs.size()
        c_in, w_in, h_in = self.input_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError(
                f"shape mismatch between expected ({self.input_shape}) and "
                f"received ({inputs.size()}) inputs."
            )
        features = self.features.view(1, c, self.num_neurons)
        bias = self.bias

        # sample the grid_locations separately per image per batch
        grid = self.sample_grid(batch_size=batch_size, sample=sample)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        outputs = F.grid_sample(inputs, grid=grid, align_corners=True)
        outputs = torch.squeeze(outputs, dim=-1) * features
        outputs = torch.sum(outputs, dim=1)
        outputs = outputs.view(batch_size, self.num_neurons)

        if bias is not None:
            outputs = outputs + bias

        return outputs
