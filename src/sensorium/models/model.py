import os
import torch
import torchinfo
import typing as t
from torch import nn
from torch.utils.data import DataLoader

from sensorium.models import shifter
from sensorium.utils import tensorboard
from sensorium.models.core import get_core
from sensorium.models.readout import Readouts


class Model(nn.Module):
    def __init__(self, args, ds: t.Dict[int, DataLoader]):
        super(Model, self).__init__()
        assert isinstance(
            args.output_shapes, dict
        ), "output_shapes must be a dictionary of mouse_id and output_shape"

        self.device = args.device
        self.input_shape = args.input_shape
        self.output_shapes = args.output_shapes
        self.use_shifter = args.include_behaviour or args.use_shifter

        self.initialize_core(args)
        self.initialize_readouts(args, ds=ds)
        self.initialize_shifter(args, ds=ds)

        self.elu = nn.ELU()

    def initialize_core(self, args):
        self.add_module(
            name="core",
            module=get_core(args)(args, input_shape=self.input_shape),
        )

    def initialize_readouts(self, args, ds: t.Dict[int, DataLoader]):
        self.add_module(
            name="readouts",
            module=Readouts(
                args,
                model=args.readout,
                input_shape=self.core.output_shape,
                output_shapes=self.output_shapes,
                ds=ds,
            ),
        )

    def initialize_shifter(self, args, ds: t.Dict[int, DataLoader]):
        if self.use_shifter:
            self.add_module(
                "shifter",
                module=shifter.MLPShifter(
                    args,
                    mouse_ids=list(ds.keys()),
                    input_channels=2,
                    hidden_features=5,
                    num_layers=3,
                ),
            )
        else:
            self.shifter = None

    def regularizer(self, mouse_id: int):
        reg = self.core.regularizer()
        reg += self.readouts.regularizer(mouse_id=mouse_id)
        if self.shifter is not None:
            reg += self.shifter.regularizer(mouse_id=mouse_id)
        return reg

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: torch.Union[int, torch.Tensor],
        pupil_center: torch.Tensor,
        activate: bool = True,
    ):
        outputs = self.core(inputs)
        shift = (
            None
            if self.shifter is None
            else self.shifter(mouse_id=mouse_id, pupil_center=pupil_center)
        )
        outputs = self.readouts(outputs, mouse_id=mouse_id, shift=shift)
        if activate:
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
        pupil_center=torch.rand(size=(args.batch_size, 2)),
    )
    with open(os.path.join(args.output_dir, "model.txt"), "w") as file:
        file.write(str(model_info))
    if args.verbose == 3:
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
        input_size=(args.batch_size, *model.core.output_shape),
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
