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
        # 1 additional patch for cls token
        num_patches = output_shape[1] * output_shape[2] + 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.positions = nn.Parameter(torch.randn(num_patches, emb_dim))
        self.output_shape = (num_patches, output_shape[0])

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


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super(FeedForward, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=dim),
        )

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        hidden_dims = emb_dim * num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(in_features=emb_dim, out_features=hidden_dims * 3)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.projection = nn.Linear(in_features=hidden_dims, out_features=emb_dim)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None):
        # split keys, queries and values in num_heads
        qkv = rearrange(
            self.qkv(inputs),
            "b n (h d qkv) -> (qkv) b h n d",
            h=self.num_heads,
            qkv=3,
        )
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        # shape (batch size, num heads, query length, key length)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)
        scale = self.dim ** (1 / 2)
        attention = self.softmax(energy) / scale
        attention = self.dropout(attention)
        # sum up over the third axis
        outputs = torch.einsum("bhal, bhlv -> bhav", attention, values)
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.projection(outputs)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, model: nn.Module, dropout: float = 0.0):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=dim)
        self.model = model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor, **kwargs):
        outputs = self.norm(inputs)
        outputs = self.model(outputs, **kwargs)
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
                    dim=emb_dim,
                    model=MultiHeadAttention(
                        emb_dim=emb_dim, num_heads=num_heads, dropout=dropout
                    ),
                    dropout=dropout,
                )
            )
            blocks.append(
                ResidualBlock(
                    dim=emb_dim,
                    model=FeedForward(dim=emb_dim, hidden_dim=mlp_dim, dropout=dropout),
                    dropout=0 if i == num_blocks - 1 else dropout,
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
        patch_size = args.patch_size
        emb_dim = args.emb_dim
        num_heads = args.num_heads
        mlp_dim = args.mlp_dim
        num_blocks = args.num_blocks
        dropout = args.dropout
        self.reg_scale = torch.tensor(args.core_reg_scale, device=args.device)

        self.patch_embedding = PatchEmbedding(
            image_shape=input_shape,
            emb_dim=emb_dim,
            patch_size=patch_size,
            stride=1 if max(input_shape) < 100 else patch_size,
        )
        self.transformer = Transformer(
            num_blocks=num_blocks,
            emb_dim=emb_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
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
            c=emb_dim,
        )
        self.output_shape = (emb_dim, latent_height, latent_width)

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
