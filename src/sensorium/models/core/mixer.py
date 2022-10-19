from .core import Core, register

import torch
from torch import nn
import typing as t
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(
    layer,
    dim: int,
    expansion_factor: float = 4,
    dropout: float = 0.0,
):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        layer(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
        layer(inner_dim, dim),
        nn.Dropout(p=dropout),
    )


@register("mixer")
class MixerCore(Core):
    def __init__(
        self,
        args,
        input_shape: t.Tuple[int, int, int],
        patch_size: int = 4,
        dim: int = 64,
        num_layers: int = 3,
        expansion_factor: int = 4,
        expansion_factor_token: float = 0.5,
        name: str = "MixerCore",
    ):
        super(MixerCore, self).__init__(args, input_shape=input_shape, name=name)
        self.patch_size = patch_size
        self.dim = dim
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.expansion_factor_token = expansion_factor_token
        self.dropout = args.dropout
        self.reg_scale = torch.tensor(args.core_reg_scale, device=args.device)

        num_channels, height, width = input_shape
        assert (height % patch_size) == 0 and (
            width % patch_size
        ) == 0, "image must be divisible by patch size"

        num_patches = (height // patch_size) * (width // patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        patch_dim = patch_size * patch_size * num_channels

        self.mixer = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(in_features=patch_dim, out_features=dim),
            *[
                nn.Sequential(
                    PreNormResidual(
                        dim,
                        FeedForward(
                            layer=chan_first,
                            dim=num_patches,
                            expansion_factor=expansion_factor,
                            dropout=args.dropout,
                        ),
                    ),
                    PreNormResidual(
                        dim,
                        FeedForward(
                            layer=chan_last,
                            dim=dim,
                            expansion_factor=expansion_factor_token,
                            dropout=args.dropout,
                        ),
                    ),
                )
                for _ in range(self.num_layers)
            ],
            nn.LayerNorm(dim)
        )

        height = 32
        self._latent_shape = (num_patches, dim)
        self._reshape_shape = (height, num_patches // height, dim)
        self.output_shape = (dim, height, num_patches // height)

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.size()[0]
        outputs = self.mixer(inputs)
        # reshape from (num patches, patch_dim) to (HWC)
        outputs = outputs.view(*(batch_size, *self._reshape_shape))
        # reorder outputs to (CHW)
        outputs = torch.permute(outputs, dims=[0, 3, 1, 2])
        return outputs
