import os
import torch
import torchinfo
import typing as t
from torch import nn
import torch.distributed
from torch.utils.data import DataLoader

from sensorium.utils import tensorboard
from sensorium.models.core import get_core
from sensorium.models.readout import Readouts
from sensorium.models.image_cropper import ImageCropper
from sensorium.models.core_shifter import CoreShifters


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
    """
    args.shift_mode:
        0 - disable shifter
        1 - shift input to core module
        2 - shift input to readout module
        3 - shift input to both core and readout module
    """

    def __init__(self, args, ds: t.Dict[int, DataLoader], name: str = "Model"):
        super(Model, self).__init__()
        assert isinstance(
            args.output_shapes, dict
        ), "output_shapes must be a dictionary of mouse_id and output_shape"
        self.name = name
        self.device = args.device
        self.input_shape = args.input_shape
        self.output_shapes = args.output_shapes
        assert args.shift_mode in (0, 1, 2, 3)
        self.shift_mode = args.shift_mode

        self.add_module(
            "image_cropper",
            module=ImageCropper(
                args,
                ds=ds,
                include_behavior=args.include_behavior,
                use_shifter=self.shift_mode in (1, 3),
            ),
        )
        self.add_module(
            name="core",
            module=get_core(args)(
                args,
                input_shape=self.image_cropper.output_shape,
            ),
        )
        if self.shift_mode in (2, 3):
            self.add_module(
                "core_shifter",
                module=CoreShifters(
                    args,
                    mouse_ids=list(ds.keys()),
                    input_channels=2,
                    hidden_features=5,
                    num_layers=3,
                ),
            )
        else:
            self.core_shifter = None
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

        self.elu = nn.ELU()

    def get_parameters(self, core_lr: float):
        # separate learning rate for core module from the rest
        params = [
            {
                "params": self.core.parameters(),
                "lr": core_lr,
                "name": "core",
            },
            {"params": self.readouts.parameters(), "name": "readouts"},
        ]
        if self.image_cropper.image_shifter is not None:
            params.append(
                {
                    "params": self.image_cropper.parameters(),
                    "name": "image_cropper",
                }
            )
        if self.core_shifter is not None:
            params.append(
                {
                    "params": self.core_shifter.parameters(),
                    "name": "core_shifter",
                }
            )
        return params

    def regularizer(self, mouse_id: int):
        reg = self.core.regularizer()
        reg += self.readouts.regularizer(mouse_id=mouse_id)
        reg += self.image_cropper.regularizer(mouse_id=mouse_id)
        if self.core_shifter is not None:
            reg += self.core_shifter.regularizer(mouse_id=mouse_id)
        return reg

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: torch.Union[int, torch.Tensor],
        pupil_center: torch.Tensor,
        behavior: torch.Tensor,
        activate: bool = True,
    ):
        images, image_grids = self.image_cropper(
            inputs,
            mouse_id=mouse_id,
            pupil_center=pupil_center,
            behavior=behavior,
        )
        outputs = self.core(images)
        shifts = None
        if self.core_shifter is not None:
            shifts = self.core_shifter(pupil_center, mouse_id=mouse_id)
        outputs = self.readouts(outputs, mouse_id=mouse_id, shifts=shifts)
        if activate:
            outputs = self.elu(outputs) + 1
        return outputs, images, image_grids


class DataParallel(nn.DataParallel):
    def __init__(self, module: Model, **kwargs):
        super(DataParallel, self).__init__(module=module, **kwargs)
        self.module = module
        self.device = module.device

    def get_parameters(self, core_lr: float):
        return self.module.get_parameters(core_lr=core_lr)


def get_model(args, ds: t.Dict[int, DataLoader], summary: tensorboard.Summary = None):
    model = Model(args, ds=ds)

    mouse_id = list(args.output_shapes.keys())[0]
    model_info = get_model_info(
        model=model,
        input_data=[
            torch.randn(args.batch_size, *model.input_shape),  # images
            mouse_id,  # mouse ID
            torch.randn(args.batch_size, 2),  # pupil centers
            torch.randn(args.batch_size, 3),  # behavior
        ],
        filename=os.path.join(args.output_dir, "model.txt"),
        summary=summary,
    )
    if args.verbose > 2:
        print(str(model_info))

    get_model_info(
        model=model.core,
        input_data=[torch.randn(args.batch_size, *model.core.input_shape)],
        filename=os.path.join(args.output_dir, "model_core.txt"),
        summary=summary,
        tag="model/trainable_parameters/core",
    )

    get_model_info(
        model=model.readouts[str(mouse_id)],
        input_data=[torch.randn(args.batch_size, *model.core.output_shape)],
        filename=os.path.join(args.output_dir, "model_readout.txt"),
        summary=summary,
        tag=f"model/trainable_parameters/Mouse{mouse_id}Readout",
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = DataParallel(model)
    model.to(args.device)

    return model
