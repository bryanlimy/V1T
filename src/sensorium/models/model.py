import os
import torch
import torchinfo
import typing as t
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from sensorium.utils import tensorboard
from sensorium.models.core import get_core
from sensorium.models.readout import Readouts


class Model(nn.Module):
    def __init__(self, args, ds: t.Dict[int, DataLoader]):
        super(Model, self).__init__()

        self.device = args.device
        self.input_shape = args.input_shape
        self.output_shapes = args.output_shapes
        assert isinstance(
            self.output_shapes, dict
        ), "output_shapes must be a dictionary of mouse_id and output_shape"

        self.initialize_core(args)
        self.initialize_readouts(args, ds=ds)

        self.elu = nn.ELU()

    def initialize_core(self, args):
        self.add_module(
            name="core", module=get_core(args)(args, input_shape=self.input_shape)
        )

    def initialize_readouts(self, args, ds: t.Dict[int, DataLoader]):
        self.add_module(
            name="readouts",
            module=Readouts(
                model=args.readout,
                input_shape=self.core.shape,
                output_shapes=self.output_shapes,
                ds=ds,
            ),
        )

    def regularizer(self):
        reg_loss = self.core.regularizer()
        for mouse_id in self.readouts.keys():
            reg_loss += self.readouts[mouse_id].regularizer()
        return reg_loss

    def forward(self, inputs: torch.Tensor, mouse_id: torch.Union[int, torch.Tensor]):
        outputs = self.core(inputs)
        outputs = self.readouts(outputs, mouse_id=mouse_id)
        outputs = self.elu(outputs) + 1
        return outputs


def get_model(args, ds: t.Dict[int, DataLoader], summary: tensorboard.Summary = None):
    model = Model(args, ds=ds)
    model.to(args.device)

    # get model summary for the first rodent
    model_info = torchinfo.summary(
        model,
        input_size=(args.batch_size, *args.input_shape),
        device=args.device,
        verbose=0,
        mouse_id=list(args.output_shapes.keys())[0],
    )
    with open(os.path.join(args.output_dir, "model.txt"), "w") as file:
        file.write(str(model_info))
    if args.verbose == 2:
        print(str(model_info))
    if summary is not None:
        summary.scalar("model/trainable_parameters", model_info.trainable_params)

    # get core model summary
    core_info = torchinfo.summary(
        model.core,
        input_size=(args.batch_size, *args.input_shape),
        device=args.device,
        verbose=0,
    )
    with open(os.path.join(args.output_dir, "model_core.txt"), "w") as file:
        file.write(str(core_info))
    if summary is not None:
        summary.scalar("model/trainable_parameters/core", core_info.trainable_params)

    # get readout model summary for the first rodent
    mouse_id = list(model.readouts.keys())[0]
    readout_info = torchinfo.summary(
        model.readouts[mouse_id],
        input_size=(args.batch_size, *model.core.shape),
        device=args.device,
        verbose=0,
    )
    with open(os.path.join(args.output_dir, "model_readout.txt"), "w") as file:
        file.write(str(readout_info))
    if summary is not None:
        summary.scalar(
            f"model/trainable_parameters/Mouse{mouse_id}Readout",
            readout_info.trainable_params,
        )

    return model
