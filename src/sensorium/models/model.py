import os
import torch
import torchinfo
from torch import nn

from sensorium.utils import tensorboard
from sensorium.models.core import get_core
from sensorium.models.readout import get_readout


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
        self.core = get_core(args)(args, input_shape=self.input_shape)

    def initialize_readouts(self, args):
        self.readouts = {}
        for mouse_id, output_shape in self.output_shapes.items():
            self.readouts[mouse_id] = get_readout(args)(
                args,
                input_shape=self.core.shape,
                output_shape=output_shape,
                name=f"Mouse{mouse_id}Readout",
            )

    def to(self, device: torch.device):
        self.core.to(device)
        for mouse_id in self.readouts.keys():
            self.readouts[mouse_id].to(device)

    def forward(self, inputs: torch.Tensor, mouse_id: torch.Union[int, torch.Tensor]):
        outputs = self.core(inputs)
        outputs = self.readouts[mouse_id](outputs)
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
