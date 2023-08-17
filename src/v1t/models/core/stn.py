from .core import register, Core

import torch
import typing as t
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from v1t.models import utils


@register("stn")
class SpatialTransformerCore(Core):
    """
    Spatial Transformer Networks
    https://arxiv.org/abs/1506.02025
    """

    def __init__(
        self,
        args,
        input_shape: t.Tuple[int, int, int],
        name: str = "SpatialTransformerCore",
    ):
        super(SpatialTransformerCore, self).__init__(
            args, input_shape=input_shape, name=name
        )
        self.register_buffer("reg_scale", torch.tensor(args.core_reg_scale))

        c, h, w = input_shape

        # Spatial transformer localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # calculate localization network output shape
        stn_shape = utils.conv2d_shape(input_shape, num_filters=8, kernel_size=7)
        stn_shape = utils.pool2d_shape(stn_shape, kernel_size=2, stride=2)
        stn_shape = utils.conv2d_shape(stn_shape, num_filters=10, kernel_size=5)
        stn_shape = utils.pool2d_shape(stn_shape, kernel_size=2, stride=2)

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
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        # CNN network
        output_shape = utils.conv2d_shape(
            input_shape,
            num_filters=args.num_filters,
            kernel_size=9,
            stride=1,
            padding=0,
        )
        output_shape = utils.conv2d_shape(
            output_shape,
            num_filters=args.num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.output_shape = output_shape

        self.cnn = nn.Sequential()

        # first layer
        layer = OrderedDict(
            {
                "conv": nn.Conv2d(
                    in_channels=c,
                    out_channels=args.num_filters,
                    kernel_size=9,
                    stride=1,
                    padding=0,
                ),
                "batchnorm": nn.BatchNorm2d(num_features=args.num_filters),
                "gelu": nn.GELU(),
                "dropout": nn.Dropout2d(p=args.dropout),
            }
        )
        self.cnn.add_module("block1", nn.Sequential(layer))
        for i in range(1, args.num_layers):
            layer = OrderedDict(
                {
                    "conv": nn.Conv2d(
                        in_channels=args.num_filters,
                        out_channels=args.num_filters,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    "batchnorm": nn.BatchNorm2d(num_features=args.num_filters),
                    "gelu": nn.GELU(),
                }
            )
            if i < args.num_layers - 1:
                layer["dropout"] = nn.Dropout2d(p=args.dropout)
            self.cnn.add_module(f"block{i+1}", nn.Sequential(layer))

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def stn(self, inputs: torch.Tensor, align_corners: bool = True):
        """Spatial transformer network forward function"""
        spatial = self.localization(inputs)
        theta = self.regressor(spatial)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, size=inputs.size(), align_corners=align_corners)
        outputs = F.grid_sample(inputs, grid=grid, align_corners=align_corners)
        return outputs

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = self.stn(inputs)
        for i, block in enumerate(self.cnn):
            outputs = block(outputs) if i == 0 else block(outputs) + outputs
        return outputs
