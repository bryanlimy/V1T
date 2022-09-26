import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from sensorium.models.neuralpredictors.readouts.base import Readout, ConfigurationError


class FullGaussian2d(Readout):
    """
    A readout using a spatial transformer layer whose positions are sampled
    from one Gaussian per neuron. Mean and covariance of that Gaussian are learned.

    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with
        `                   `init_sigma` when `gauss_type` is 'isotropic' or
                            'uncorrelated'.
                            When `gauss_type='full'` initialise the square
                            root of the covariance matrix with
                            Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the
                            batch separately [default: True as it decreases
                            convergence time and performs just as well]
        align_corners (bool): Keyword argument to gridsample for bilinear
                            interpolation. It changed behavior in PyTorch 1.3.
                            The default of align_corners = True is setting the
                            behavior to pre PyTorch 1.3 functionality for
                            comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic',
                        'uncorrelated', or 'full' (default).
        grid_mean_predictor (dict): Parameters for a predictor of the mean
                                    grid locations. Has to have a form like
                                    {
                                        'hidden_layers':0,
                                        'hidden_features':20,
                                        'final_tanh': False,
                                    }
        shared_features (dict): Used when the feature vectors are shared
                                (within readout between neurons) or between
                                this readout and other readouts. Has to be a
                                dictionary of the form
                                {
                                    'match_ids': (numpy.array),
                                    'shared_features': torch.nn.Parameter or None
                                }
                                The match_ids are used to match things that
                                should be shared within or across scans.
                                If `shared_features` is None, this readout will
                                create its own features. If it is set to a
                                feature Parameter of another readout, it will
                                replace the features of this readout. It will
                                be accessed in increasing order of the sorted
                                unique match_ids.
                                For instance, if match_ids=[2,0,0,1], there
                                should be 3 features in order [0,1,2]. When
                                this readout creates features, it will do so
                                in that order.
        shared_grid (dict): Like `shared_features`. Use dictionary like
                            {
                                'match_ids': (numpy.array),
                                'shared_grid': torch.nn.Parameter or None
                            }
                            See documentation of `shared_features` for specification.
        source_grid (numpy.array): Source grid for the grid_mean_predictor.
                                Needs to be of size
                                (neurons, grid_mean_predictor[input_dimensions])
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_mu_range=0.1,
        init_sigma=1,
        batch_sample=True,
        align_corners=True,
        gauss_type="full",
        grid_mean_predictor=None,
        shared_features=None,
        shared_grid=None,
        source_grid=None,
        mean_activity=None,
        feature_reg_weight=1.0,
        gamma_readout=None,  # deprecated, use feature_reg_weight instead
        **kwargs,
    ):

        super().__init__()
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(
            feature_reg_weight, gamma_readout
        )
        self.mean_activity = mean_activity
        # determines whether the Gaussian is isotropic or not
        self.gauss_type = gauss_type

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError(
                "either init_mu_range doesn't belong to [0.0, 1.0] or "
                "init_sigma_range is non-positive"
            )

        # store statistics about the images and neurons
        self.in_shape = in_shape
        self.outdims = outdims

        # sample a different location per example
        self.batch_sample = batch_sample

        # position grid shape
        self.grid_shape = (1, outdims, 1, 2)

        # the grid can be predicted from another grid
        self._predicted_grid = False
        self._shared_grid = False
        self._original_grid = not self._predicted_grid

        if grid_mean_predictor is None and shared_grid is None:
            # mean location of gaussian for each neuron
            self._mu = Parameter(torch.Tensor(*self.grid_shape))
        elif grid_mean_predictor is not None and shared_grid is not None:
            raise ConfigurationError(
                "Shared grid_mean_predictor and shared_grid_mean cannot both be set"
            )
        elif grid_mean_predictor is not None:
            self.init_grid_predictor(source_grid=source_grid, **grid_mean_predictor)
        elif shared_grid is not None:
            self.initialize_shared_grid(**(shared_grid or {}))

        if gauss_type == "full":
            self.sigma_shape = (1, outdims, 2, 2)
        elif gauss_type == "uncorrelated":
            self.sigma_shape = (1, outdims, 1, 2)
        elif gauss_type == "isotropic":
            self.sigma_shape = (1, outdims, 1, 1)
        else:
            raise ValueError(f'gauss_type "{gauss_type}" not known')

        self.init_sigma = init_sigma
        self.sigma = Parameter(
            torch.Tensor(*self.sigma_shape)
        )  # standard deviation for gaussian for each neuron

        self.initialize_features(**(shared_features or {}))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.align_corners = align_corners
        self.initialize(mean_activity)

    @property
    def shared_features(self):
        return self._features

    @property
    def shared_grid(self):
        return self._mu

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[..., self.feature_sharing_index]
        else:
            return self._features

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    @property
    def mu_dispersion(self):
        """
        Returns the standard deviation of the learned positions.
        Is used as a regularizer to push neurons to learn similar positions.

        Returns:
            mu_dispersion(float): average dispersion of the mean 2d-position
        """

        return self._mu.squeeze().std(0).sum()

    def feature_l1(self, reduction="sum", average=None):
        """
        Returns l1 regularization term for features.
        Args:
            average(bool): Deprecated (see reduction) if True, use mean of
                            weights for regularization
            reduction(str): Specifies the reduction to apply to the output:
                            'none' | 'mean' | 'sum'
        """
        if self._original_features:
            return self.apply_reduction(
                self.features.abs(), reduction=reduction, average=average
            )
        else:
            return 0

    def regularizer(self, reduction="sum", average=None):
        return (
            self.feature_l1(reduction=reduction, average=average)
            * self.feature_reg_weight
        )

    @property
    def mu(self):
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
        elif self._shared_grid:
            if self._original_grid:
                return self._mu[:, self.grid_sharing_index, ...]
            else:
                return self.mu_transform(self._mu.squeeze())[
                    self.grid_sharing_index
                ].view(*self.grid_shape)
        else:
            return self._mu

    def sample_grid(self, batch_size, sample=None):
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
            norm = self.mu.new(
                *grid_shape
            ).zero_()  # for consistency and CUDA capability

        if self.gauss_type != "full":
            # grid locations in feature space sampled randomly around the mean self.mu
            return torch.clamp(norm * self.sigma + self.mu, min=-1, max=1)
        else:
            # grid locations in feature space sampled randomly around the mean self.mu
            return torch.clamp(
                torch.einsum("ancd,bnid->bnic", self.sigma, norm) + self.mu,
                min=-1,
                max=1,
            )

    def init_grid_predictor(
        self, source_grid, hidden_features=20, hidden_layers=0, final_tanh=False
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

        if final_tanh:
            layers.append(nn.Tanh())
        self.mu_transform = nn.Sequential(*layers)

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer(
            "source_grid", torch.from_numpy(source_grid.astype(np.float32))
        )
        self._predicted_grid = True

    def initialize(self, mean_activity=None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with
        the features weights
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gauss_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        self._features.data.fill_(1 / self.in_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes
        whether this instance of the FullGaussian2d learns the original
        features (True) or if it uses a copy of the features from another
        instance of FullGaussian2d via the `shared_features` (False). If it
        uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    1,
                    c,
                    1,
                    n_match_ids,
                ), f"shared features need to have shape (1, {c}, 1, {n_match_ids})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(
                    torch.Tensor(1, c, 1, n_match_ids)
                )  # feature weights for each channel of the core
            self.scales = Parameter(
                torch.Tensor(1, 1, 1, self.outdims)
            )  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(
                torch.Tensor(1, c, 1, self.outdims)
            )  # feature weights for each channel of the core
            self._shared_features = False

    def initialize_shared_grid(self, match_ids=None, shared_grid=None):
        c, w, h = self.in_shape

        if match_ids is None:
            raise ConfigurationError("match_ids must be set for sharing grid")
        assert self.outdims == len(
            match_ids
        ), "There must be one match ID per output dimension"

        n_match_ids = len(np.unique(match_ids))
        if shared_grid is not None:
            assert shared_grid.shape == (
                1,
                n_match_ids,
                1,
                2,
            ), f"shared grid needs to have shape (1, {n_match_ids}, 1, 2)"
            self._mu = shared_grid
            self._original_grid = False
            self.mu_transform = nn.Linear(2, 2)
            self.mu_transform.bias.data.fill_(0.0)
            self.mu_transform.weight.data = torch.eye(2)
        else:
            self._mu = Parameter(
                torch.Tensor(1, n_match_ids, 1, 2)
            )  # feature weights for each channel of the core
        _, sharing_idx = np.unique(match_ids, return_inverse=True)
        self.register_buffer("grid_sharing_index", torch.from_numpy(sharing_idx))
        self._shared_grid = True

    def forward(self, x, sample=None, shift=None, out_idx=None, **kwargs):
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
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            warnings.warn(
                "the specified feature map dimension is not the readout's "
                "expected input dimension"
            )
        feat = self.features.view(1, c, self.outdims)
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(
                batch_size=N, sample=sample
            )  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(
                N, outdims, 1, 2
            )

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.gauss_type + " "
        r += (
            self.__class__.__name__
            + " ("
            + "{} x {} x {}".format(c, w, h)
            + " -> "
            + str(self.outdims)
            + ")"
        )
        if self.bias is not None:
            r += " with bias"
        if self._shared_features:
            r += ", with {} features".format(
                "original" if self._original_features else "shared"
            )

        if self._predicted_grid:
            r += ", with predicted grid"
        if self._shared_grid:
            r += ", with {} grid".format(
                "original" if self._original_grid else "shared"
            )

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r
