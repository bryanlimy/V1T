import os
import torch
import torchinfo
from torch import nn

from sensorium.utils import tensorboard
from sensorium.models.core import get_core
from sensorium.models.readout import Readouts


class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()

        self.device = args.device
        self.input_shape = args.input_shape
        self.output_shapes = args.output_shapes
        assert isinstance(
            self.output_shapes, dict
        ), "output_shapes must be a dictionary of mouse_id and output_shape"

        self.initialize_core(args)
        self.initialize_readouts(args)

    def initialize_core(self, args):
        self.add_module(
            name="core", module=get_core(args)(args, input_shape=self.input_shape)
        )

    def initialize_readouts(self, args):
        self.add_module(
            name="readouts",
            module=Readouts(
                model=args.readout,
                input_shape=self.core.shape,
                output_shapes=self.output_shapes,
            ),
        )

    def forward(self, inputs: torch.Tensor, mouse_id: torch.Union[int, torch.Tensor]):
        outputs = self.core(inputs)
        outputs = self.readouts(outputs, mouse_id=mouse_id)
        return outputs


def get_model(args, summary: tensorboard.Summary = None):
    model = BasicModel(args)
    model.to(args.device)

    # get model summary and write to args.output_dir
    core_info = torchinfo.summary(
        model.core,
        input_size=(args.batch_size, *args.input_shape),
        device=args.device,
        verbose=0,
    )
    with open(os.path.join(args.output_dir, "model_core.txt"), "w") as file:
        file.write(str(core_info))
    if args.verbose == 2:
        print(str(core_info))
    if summary is not None:
        summary.scalar("model/core/trainable_parameters", core_info.trainable_params)

    mouse_id = list(model.readouts.keys())[0]
    readout_info = torchinfo.summary(
        model.readouts[mouse_id],
        input_size=(args.batch_size, *model.core.shape),
        device=args.device,
        verbose=0,
    )
    with open(os.path.join(args.output_dir, "model_readout.txt"), "w") as file:
        file.write(str(readout_info))
    if args.verbose == 2:
        print(str(readout_info))
    if summary is not None:
        summary.scalar(
            "model/readout/trainable_parameters", readout_info.trainable_params
        )

    return model
