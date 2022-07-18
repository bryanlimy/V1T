from .registry import register

import torch
from torch import nn


@register("random")
def get_model(args):
    return RandomModel(args)


class RandomModel(nn.Module):
    def __init__(self, args):
        super(RandomModel, self).__init__()

        self.device = args.device
        self.input_shape = args.input_shape
        self.output_shape = args.output_shape
        self.weights = nn.Parameter(torch.randn(1), requires_grad=True)
        self.elu = nn.ELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = (inputs.shape[0], *self.output_shape)
        outputs = torch.rand(shape, device=self.device)
        outputs += self.weights - self.weights.detach()
        return self.elu(outputs) + 1
