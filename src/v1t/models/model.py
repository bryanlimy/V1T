import os
import wandb
import torch
import torchinfo
import typing as t
from torch import nn
import torch.distributed
from torch.utils.data import DataLoader


from v1t.utils import tensorboard
from v1t.models.core import get_core
from v1t.models.readout import Readouts
from v1t.models.core_shifter import CoreShifters
from v1t.models.image_cropper import ImageCropper
from v1t.models.utils import ELU1, load_pretrain_core


def get_model_info(
    model: nn.Module,
    input_data: t.Union[torch.Tensor, t.Sequence[t.Any], t.Mapping[str, t.Any]],
    mouse_id: str = None,
    filename: str = None,
    summary: tensorboard.Summary = None,
    device: torch.device = "cpu",
    tag: str = "model/trainable_parameters",
):
    args = {
        "model": model,
        "input_data": input_data,
        "depth": 5,
        "device": device,
        "verbose": 0,
    }
    if mouse_id is not None:
        args["mouse_id"] = mouse_id
    model_info = torchinfo.summary(**args)
    if filename is not None:
        with open(filename, "w") as file:
            file.write(str(model_info))
    if summary is not None:
        summary.scalar(tag, model_info.trainable_params)
    return model_info


class Model(nn.Module):
    """
    shift mode:
        0 - disable shifter
        1 - shift input to core module
        2 - shift input to readout module
        3 - shift input to both core and readout module
        4 - shift_mode=3 and provide both behavior and pupil center to cropper
    """

    def __init__(self, args, ds: t.Dict[str, DataLoader], name: str = "Model"):
        super(Model, self).__init__()
        assert isinstance(
            args.output_shapes, dict
        ), "output_shapes must be a dictionary of mouse_id and output_shape"
        self.name = name
        self.input_shape = args.input_shape
        self.output_shapes = args.output_shapes
        self.shift_mode = args.shift_mode

        self.add_module(
            "image_cropper",
            module=ImageCropper(args, ds=ds),
        )
        self.add_module(
            name="core",
            module=get_core(args)(
                args,
                input_shape=self.image_cropper.output_shape,
            ),
        )
        if self.shift_mode in (2, 3, 4):
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

        self.elu1 = ELU1()

    def get_parameters(self, core_lr: float):
        # separate learning rate for core module from the rest
        params = []
        if self.core.requires_grad_:
            params.append(
                {
                    "params": self.core.parameters(),
                    "lr": core_lr,
                    "name": "core",
                }
            )
        if self.readouts.requires_grad_:
            params.append({"params": self.readouts.parameters(), "name": "readouts"})
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

    def regularizer(self, mouse_id: str):
        reg = self.core.regularizer()
        reg += self.readouts.regularizer(mouse_id=mouse_id)
        reg += self.image_cropper.regularizer(mouse_id=mouse_id)
        if self.core_shifter is not None:
            reg += self.core_shifter.regularizer(mouse_id=mouse_id)
        return reg

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        activate: bool = True,
    ):
        images, image_grids = self.image_cropper(
            inputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        outputs = self.core(
            images,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        shifts = None
        if self.core_shifter is not None:
            shifts = self.core_shifter(pupil_centers, mouse_id=mouse_id)
        outputs = self.readouts(outputs, mouse_id=mouse_id, shifts=shifts)
        if activate:
            outputs = self.elu1(outputs)
        return outputs, images, image_grids


class DataParallel(nn.DataParallel):
    def __init__(self, module: Model, **kwargs):
        super(DataParallel, self).__init__(module=module, **kwargs)
        self.module = module
        self.input_shape = module.input_shape
        self.output_shapes = self.module.output_shapes

    def get_parameters(self, core_lr: float):
        return self.module.get_parameters(core_lr)

    def regularizer(self, mouse_id: int):
        return self.module.regularizer(mouse_id)


def get_model(args, ds: t.Dict[str, DataLoader], summary: tensorboard.Summary = None):
    model = Model(args, ds=ds)

    if hasattr(args, "pretrain_core") and args.pretrain_core:
        load_pretrain_core(args, model=model, device=args.device)
        model.core.requires_grad_(False)
        if args.verbose:
            print("Freeze pretrained core")

    # get model info
    mouse_id = args.mouse_ids[0]
    batch_size = args.micro_batch_size
    random_input = lambda size: torch.rand(*size)
    model_info = get_model_info(
        model=model,
        input_data={
            "inputs": random_input((batch_size, *model.input_shape)),
            "behaviors": random_input((batch_size, 3)),
            "pupil_centers": random_input((batch_size, 2)),
        },
        mouse_id=mouse_id,
        filename=os.path.join(args.output_dir, "model.txt"),
        summary=summary,
    )
    if args.verbose > 2:
        print(str(model_info))
    if args.use_wandb:
        wandb.log({"trainable_params": model_info.trainable_params}, step=0)

    # get core info
    get_model_info(
        model=model.core,
        input_data={
            "inputs": random_input((batch_size, *model.core.input_shape)),
            "behaviors": random_input((batch_size, 3)),
            "pupil_centers": random_input((batch_size, 2)),
        },
        mouse_id=mouse_id,
        filename=os.path.join(args.output_dir, "model_core.txt"),
        summary=summary,
        tag="model/trainable_parameters/core",
    )
    # get readout summary
    get_model_info(
        model=model.readouts[mouse_id],
        input_data={"inputs": random_input((batch_size, *model.core.output_shape))},
        filename=os.path.join(args.output_dir, "model_readout.txt"),
        summary=summary,
        tag=f"model/trainable_parameters/Mouse{mouse_id}Readout",
    )

    model.to(args.device)
    return model
