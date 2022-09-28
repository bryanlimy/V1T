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
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True),
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
        self.fc_loc = nn.Sequential(
            nn.Linear(in_features=int(np.prod(stn_shape)), out_features=32),
            nn.ReLU(True),
            nn.Linear(in_features=32, out_features=3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout2d = nn.Dropout2d(p=0.25)

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
        theta = self.fc_loc(spatial)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, inputs.size())
        outputs = F.grid_sample(inputs, grid=grid, align_corners=True)
        return outputs

    def forward(self, inputs: torch.Tensor):
        # transform the input
        outputs = self.stn(inputs)

        # Perform the usual forward pass
        outputs = self.conv1(outputs)
        outputs = F.max_pool2d(outputs, kernel_size=2)
        outputs = F.gelu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2d(outputs)
        outputs = F.max_pool2d(outputs, kernel_size=2)
        outputs = F.gelu(outputs)
        return outputs
