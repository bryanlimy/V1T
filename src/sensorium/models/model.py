import os
import torch
import torchinfo
import typing as t
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from sensorium.models import shifter
from sensorium.utils import tensorboard
from sensorium.models.core import get_core
from sensorium.models.readout import Readouts


def get_model_info(
    model: nn.Module,
    input_data: t.Union[torch.Tensor, t.Sequence[t.Any], t.Mapping[str, t.Any]],
    filename: str = None,
    summary: tensorboard.Summary = None,
    tag: str = "model/trainable_parameters",
):
    model_info = torchinfo.summary(
        model,
        input_data=input_data,
        depth=5,
        device=model.device,
        verbose=0,
    )
    if filename is not None:
        with open(filename, "w") as file:
            file.write(str(model_info))
    if summary is not None:
        summary.scalar(tag, model_info.trainable_params)
    return model_info


class Model(nn.Module):
    def __init__(self, args, ds: t.Dict[int, DataLoader], name: str = "Model"):
        super(Model, self).__init__()
        assert isinstance(
            args.output_shapes, dict
        ), "output_shapes must be a dictionary of mouse_id and output_shape"
        self.name = name
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

    model_info = get_model_info(
        model=model,
        input_data=[
            torch.randn(args.batch_size, *args.input_shape),  # images
            list(args.output_shapes.keys())[0],  # mouse ID
            torch.randn(args.batch_size, 2),  # pupil centers
        ],
        filename=os.path.join(args.output_dir, "model.txt"),
        summary=summary,
    )
    if args.verbose > 2:
        print(str(model_info))

    get_model_info(
        model=model.core,
        input_data=[torch.randn(args.batch_size, *args.input_shape)],
        filename=os.path.join(args.output_dir, "model_core.txt"),
        summary=summary,
        tag="model/trainable_parameters/core",
    )

    mouse_id = list(model.readouts.keys())[0]
    get_model_info(
        model=model.readouts[mouse_id],
        input_data=[torch.randn(args.batch_size, *model.core.output_shape)],
        filename=os.path.join(args.output_dir, "model_readout.txt"),
        summary=summary,
        tag=f"model/trainable_parameters/Mouse{mouse_id}Readout",
    )

    return model
