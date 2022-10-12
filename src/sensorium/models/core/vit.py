from .core import register, Core

import math
import torch
import typing as t
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from sensorium.models import utils as model_utils


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, inputs: torch.Tensor, **kwargs):
        return self.fn(self.norm(inputs), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=dim),
        )

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.to_qkv = nn.Linear(in_features=dim, out_features=inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(in_features=inner_dim, out_features=dim),
                nn.Dropout(p=dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor):
        qkv = self.to_qkv(inputs).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        outputs = torch.matmul(attn, v)
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        return self.to_out(outputs)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super(Transformer, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim=dim,
                            fn=Attention(
                                dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(
                            dim=dim,
                            fn=FeedForward(
                                dim=dim, hidden_dim=mlp_dim, dropout=dropout
                            ),
                        ),
                    ]
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        for attention, feedforward in self.layers:
            outputs = attention(outputs) + outputs
            outputs = feedforward(outputs) + outputs
        return outputs


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_shape: t.Tuple[int, int, int],
        emb_dim: int,
        patch_size: int,
        stride: int = 1,
    ):
        super(PatchEmbedding, self).__init__()
        c, h, w = image_shape
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=emb_dim,
                kernel_size=patch_size,
                stride=stride,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        output_shape = model_utils.conv2d_shape(
            input_shape=image_shape,
            num_filters=emb_dim,
            kernel_size=patch_size,
            stride=stride,
        )
        num_patches = output_shape[1] * output_shape[2] + 1
        self.output_shape = (num_patches, output_shape[0])

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.positions = nn.Parameter(torch.randn((num_patches, emb_dim)))

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.size(0)
        patches = self.projection(inputs)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=batch_size)
        patches = torch.cat([cls_tokens, patches], dim=1)
        patches += self.positions
        return patches


def find_shape(num_patches: int):
    num1 = math.floor(math.sqrt(num_patches))
    while num_patches % num1 != 0:
        num1 -= 1
    num2 = num_patches // num1
    return num1, num2


class UpsampleConv(nn.Module):
    def __init__(self, input_shape: t.Tuple[int, int]):
        super(UpsampleConv, self).__init__()
        l_in, c = input_shape
        latent_dim = self.next_perfect_square(l_in)
        l_out = latent_dim * latent_dim
        kernel_size = self.get_kernel_size(l_in=l_in, l_out=l_out)

        self.upsample = nn.Sequential(
            Rearrange("b l c -> b c l"),
            nn.ConvTranspose1d(
                in_channels=c,
                out_channels=c,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                output_padding=0,
                dilation=1,
            ),
            Rearrange("b c (lh lw) -> b c lh lw", lh=latent_dim, lw=latent_dim),
            nn.ELU(),
        )
        self.output_shape = (c, latent_dim, latent_dim)

    def next_perfect_square(self, number: int):
        return math.floor(math.sqrt(number)) + 1

    def get_kernel_size(self, l_in: int, l_out: int):
        return l_out - l_in + 1

    def forward(self, inputs: torch.Tensor):
        return self.upsample(inputs)


@register("vit")
class ViTCore(Core):
    def __init__(
        self,
        args,
        input_shape: t.Tuple[int, int, int],
        name: str = "ViTCore",
    ):
        super(ViTCore, self).__init__(args, input_shape=input_shape, name=name)
        patch_size = args.patch_size
        emb_dim = args.emb_dim
        heads = args.num_heads
        mlp_dim = args.mlp_dim
        num_layers = args.num_layers
        dim_head = args.dim_head
        dropout = args.dropout

        self.patch_embedding = PatchEmbedding(
            image_shape=input_shape,
            emb_dim=emb_dim,
            patch_size=patch_size,
            stride=patch_size if max(input_shape) > 100 else 1,
        )
        output_shape = self.patch_embedding.output_shape

        self.transformer = Transformer(
            dim=emb_dim,
            num_layers=num_layers,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.upsample = UpsampleConv(input_shape=output_shape)
        self.output_shape = self.upsample.output_shape

    def forward(self, inputs: torch.Tensor):
        outputs = self.patch_embedding(inputs)
        outputs = self.transformer(outputs)
        outputs = self.upsample(outputs)
        return outputs
