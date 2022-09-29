from .core import register, Core

import torch
import typing as t
from torch import nn
import torch.nn.functional as F

import numpy as np

from sensorium.models import utils


@register("stn")
class SpatialTransformerCore(Core):
    """
    Spatial Transformer Networks
    https://arxiv.org/abs/1506.02025
    """

    def __init__(
        self, args, input_shape: t.Tuple[int], name: str = "SpatialTransformerCore"
    ):
        super(SpatialTransformerCore, self).__init__(
            args, input_shape=input_shape, name=name
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        stn_shape = utils.conv2d_output_shape(
            input_shape=input_shape, num_filters=8, kernel_size=7
        )
        stn_shape = (stn_shape[0], stn_shape[1] // 2, stn_shape[2] // 2)
        stn_shape = utils.conv2d_output_shape(
            input_shape=stn_shape, num_filters=10, kernel_size=5
        )
        stn_shape = (stn_shape[0], stn_shape[1] // 2, stn_shape[2] // 2)

        self.flatten = nn.Flatten()

        # Regressor for the 3 * 2 affine matrix
        self.localization_feedforward = nn.Sequential(
            nn.Linear(in_features=int(np.prod(stn_shape)), out_features=32),
            nn.ReLU(True),
            nn.Linear(in_features=32, out_features=3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.localization_feedforward[2].weight.data.zero_()
        self.localization_feedforward[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        )

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.GELU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.Dropout2d(p=args.dropout),
            nn.MaxPool2d(kernel_size=2),
            nn.GELU(),
        )

        output_shape = utils.conv2d_output_shape(
            input_shape=input_shape, num_filters=10, kernel_size=5
        )
        output_shape = (output_shape[0], output_shape[1] // 2, output_shape[2] // 2)
        output_shape = utils.conv2d_output_shape(
            input_shape=output_shape, num_filters=20, kernel_size=5
        )
        output_shape = (output_shape[0], output_shape[1] // 2, output_shape[2] // 2)
        self._output_shape = output_shape

    # Spatial transformer network forward function
    def stn(self, inputs: torch.Tensor):
        spatial = self.localization(inputs)
        spatial = self.flatten(spatial)
        theta = self.localization_feedforward(spatial)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, size=inputs.size(), align_corners=False)
        outputs = F.grid_sample(inputs, grid=grid, align_corners=False)
        return outputs

    def forward(self, inputs: torch.Tensor):
        outputs = self.stn(inputs)
        outputs = self.feedforward(outputs)
        return outputs
