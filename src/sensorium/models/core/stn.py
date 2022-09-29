from .core import register, Core

import torch
import typing as t
import numpy as np
from torch import nn
import torch.nn.functional as F

from sensorium.models import utils


@register("stn")
class SpatialTransformerCore(Core):
    """
    Spatial Transformer Networks
    https://arxiv.org/abs/1506.02025
    """

    def __init__(
        self,
        args,
        input_shape: t.Tuple[int],
        name: str = "SpatialTransformerCore",
    ):
        super(SpatialTransformerCore, self).__init__(
            args, input_shape=input_shape, name=name
        )

        # Spatial transformer localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # calculate localization network output shape
        stn_shape = utils.conv2d_output_shape(
            input_shape=input_shape, num_filters=8, kernel_size=7
        )
        stn_shape = (stn_shape[0], stn_shape[1] // 2, stn_shape[2] // 2)
        stn_shape = utils.conv2d_output_shape(
            input_shape=stn_shape, num_filters=10, kernel_size=5
        )
        stn_shape = (stn_shape[0], stn_shape[1] // 2, stn_shape[2] // 2)

        # Regressor for the 3 * 2 affine matrix
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=int(np.prod(stn_shape)), out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.regressor[3].weight.data.zero_()
        self.regressor[3].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        )

        # Feedforward network
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=args.num_filters,
                kernel_size=5,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.GELU(),
            nn.Dropout2d(p=args.dropout),
            nn.Conv2d(
                in_channels=args.num_filters,
                out_channels=args.num_filters * 2,
                kernel_size=5,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.GELU(),
        )

        # calculate feedforward network output shape
        output_shape = utils.conv2d_output_shape(
            input_shape=input_shape, num_filters=args.num_filters, kernel_size=5
        )
        output_shape = (output_shape[0], output_shape[1] // 2, output_shape[2] // 2)
        output_shape = utils.conv2d_output_shape(
            input_shape=output_shape, num_filters=args.num_filters * 2, kernel_size=5
        )
        output_shape = (output_shape[0], output_shape[1] // 2, output_shape[2] // 2)
        self._output_shape = output_shape

    def stn(self, inputs: torch.Tensor, align_corners: bool = True):
        """Spatial transformer network forward function"""
        spatial = self.localization(inputs)
        theta = self.regressor(spatial)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, size=inputs.size(), align_corners=align_corners)
        outputs = F.grid_sample(inputs, grid=grid, align_corners=align_corners)
        return outputs

    def forward(self, inputs: torch.Tensor):
        outputs = self.stn(inputs)
        outputs = self.network(outputs)
        return outputs
