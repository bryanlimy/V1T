from .core import register, Core

import math
import torch
import typing as t
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sensorium.models.utils as model_utils


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_shape: t.Tuple[int, int, int],
        emb_dim: int,
        patch_size: int,
        stride: int = 1,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        c, h, w = image_shape
        # tensor dimension after unfolding inputs
        # (batch_size, (patch_size * patch_size * c), (new_h * new_w))
        new_h, new_w = self.latent_size(h), self.latent_size(w)
        num_patches = new_h * new_w
        patch_dim = patch_size * patch_size * c
        self.projection = nn.Sequential(
            nn.Unfold(kernel_size=self.patch_size, stride=stride),
            Rearrange("b c n -> b n c"),
            nn.Linear(in_features=patch_dim, out_features=emb_dim),
        )
        num_patches = num_patches + 1  # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.positions = nn.Parameter(torch.randn(num_patches, emb_dim))
        self.output_shape = (num_patches, emb_dim)

    def latent_size(self, in_size: int):
        """Calculate the spatial dimension of the outputs after unfold"""
        return int(((in_size - (self.patch_size - 1) - 1) / self.stride) + 1)

    @property
    def num_patches(self):
        return self.output_shape[0]

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.size(0)
        patches = self.projection(inputs)
        cls_tokens = repeat(self.cls_token, "1 1 c -> b 1 c", b=batch_size)
        patches = torch.cat([cls_tokens, patches], dim=1)
        patches += self.positions
        return patches


def feedforward(dim: int, hidden_dim: int, dropout: float = 0.0):
    return nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=hidden_dim, out_features=dim),
    )


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        inner_dim = emb_dim * num_heads
        self.num_heads = num_heads
        self.register_buffer("scale", torch.tensor(emb_dim**-0.5))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(
            in_features=emb_dim, out_features=inner_dim * 3, bias=False
        )
        self.projection = nn.Linear(in_features=inner_dim, out_features=emb_dim)

    def forward(self, inputs: torch.Tensor):
        qkv = self.to_qkv(inputs)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attention = self.softmax(dots)
        attention = self.dropout(attention)

        outputs = torch.matmul(attention, v)
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.projection(outputs)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, emb_dim: int, model: nn.Module, dropout: float = 0.0):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=emb_dim)
        self.model = model
        self.dropout = None if dropout is None else nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor, **kwargs):
        outputs = self.norm(inputs)
        outputs = self.model(outputs, **kwargs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        return outputs + inputs


class Transformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        emb_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super(Transformer, self).__init__()
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResidualBlock(
                    emb_dim=emb_dim,
                    model=MultiHeadAttention(
                        emb_dim=emb_dim, num_heads=num_heads, dropout=dropout
                    ),
                    dropout=dropout,
                )
            )
            blocks.append(
                ResidualBlock(
                    emb_dim=emb_dim,
                    model=feedforward(dim=emb_dim, hidden_dim=mlp_dim, dropout=dropout),
                    dropout=None if i == num_blocks - 1 else dropout,
                )
            )
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)


@register("vit")
class ViTCore(Core):
    def __init__(
        self,
        args,
        input_shape: t.Tuple[int, int, int],
        name: str = "ViTCore",
    ):
        super(ViTCore, self).__init__(args, input_shape=input_shape, name=name)
        self.reg_scale = torch.tensor(args.core_reg_scale, device=args.device)

        self.patch_embedding = PatchEmbedding(
            image_shape=input_shape,
            emb_dim=args.emb_dim,
            patch_size=args.patch_size,
            stride=1 if max(input_shape) < 100 else args.patch_size,
        )
        self.transformer = Transformer(
            num_blocks=args.num_blocks,
            emb_dim=args.emb_dim,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
        )
        # calculate latent height and width based on num_patches
        latent_height, latent_width = self.find_shape(
            num_patches=self.patch_embedding.num_patches
        )
        # reshape transformer output from (batch_size, num_patches, channels)
        # to (batch_size, channel, latent height, latent width)
        self.rearrange = Rearrange(
            "b (lh lw) c -> b c lh lw",
            lh=latent_height,
            lw=latent_width,
            c=args.emb_dim,
        )
        self.output_shape = (args.emb_dim, latent_height, latent_width)

    @staticmethod
    def find_shape(num_patches: int):
        """Find the height and width values that can make up the number of patches"""
        height = math.floor(math.sqrt(num_patches))
        while num_patches % height != 0:
            height -= 1
        width = num_patches // height
        return height, width

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        outputs = self.patch_embedding(inputs)
        outputs = self.transformer(outputs)
        outputs = self.rearrange(outputs)
        return outputs
