import torch
import typing as t
from torch import nn


class DropPath(nn.Module):
    """
    Stochastic depth for regularization https://arxiv.org/abs/1603.09382
    Reference:
    - https://github.com/aanna0701/SPT_LSA_ViT/blob/main/utils/drop_path.py
    - https://github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dropout: float = 0.0):
        super(DropPath, self).__init__()
        self.register_buffer("keep_prop", torch.tensor(1 - dropout))

    def forward(self, inputs: torch.Tensor):
        if self.keep_prop == 1 or not self.training:
            return inputs
        shape = (inputs.size(0),) + (1,) * (inputs.ndim - 1)
        random_tensor = torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        random_tensor = torch.floor(self.keep_prop + random_tensor)
        outputs = (inputs / self.keep_prop) * random_tensor
        return outputs
