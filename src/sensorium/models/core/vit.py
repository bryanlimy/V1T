from .core import register, Core

import math
import torch
import typing as t
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Image2Patches(nn.Module):
    def __init__(
        self,
        image_shape: t.Tuple[int, int, int],
        patch_size: int,
        stride: int,
        emb_dim: int,
        dropout: float = 0.0,
    ):
        super(Image2Patches, self).__init__()
        c, h, w = image_shape
        self.patch_size = patch_size
        self.stride = stride
        new_h = int(((image_shape[1] - (patch_size - 1) - 1) / stride) + 1)
        new_w = int(((image_shape[2] - (patch_size - 1) - 1) / stride) + 1)
        self.rearrange = Rearrange(
            "b c h w p1 p2 -> b (h w) (p1 p2 c)",
            h=new_h,
            w=new_w,
            p1=patch_size,
            p2=patch_size,
        )
        patch_dim = patch_size * patch_size * c
        num_patches = new_h * new_w

        self.linear = nn.Linear(in_features=patch_dim, out_features=emb_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.dropout = nn.Dropout(p=dropout)
        # num_patches += 1
        self.num_patches = num_patches

    def forward(self, inputs: torch.Tensor):
        patches = inputs.unfold(
            dimension=2, size=self.patch_size, step=self.stride
        ).unfold(dimension=3, size=self.patch_size, step=self.stride)
        patches = self.rearrange(patches)
        b, n, _ = patches.shape

        outputs = self.linear(patches)

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        outputs = torch.cat((cls_tokens, outputs), dim=1)
        outputs += self.pos_embedding[:, : n + 1]
        outputs = self.dropout(outputs)

        return outputs


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
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)


class Attention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super(Attention, self).__init__()
        inner_dim = emb_dim * num_heads

        self.num_heads = num_heads
        self.scale = emb_dim**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.to_qkv = nn.Linear(
            in_features=emb_dim, out_features=inner_dim * 3, bias=False
        )

        self.projection = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor):
        qkv = self.to_qkv(inputs).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        outputs = torch.matmul(attn, v)
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.projection(outputs)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim=emb_dim,
                            fn=Attention(
                                emb_dim=emb_dim,
                                num_heads=num_heads,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(
                            dim=emb_dim,
                            fn=FeedForward(
                                dim=emb_dim, hidden_dim=mlp_dim, dropout=dropout
                            ),
                        ),
                    ]
                )
            )

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        for attn, ff in self.layers:
            outputs = attn(outputs) + outputs
            outputs = ff(outputs) + outputs
        return outputs


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

        self.patch_embedding = Image2Patches(
            image_shape=input_shape,
            patch_size=args.patch_size,
            stride=1 if max(input_shape) < 100 else args.patch_size,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
        )
        num_patches = self.patch_embedding.num_patches

        self.transformer = Transformer(
            emb_dim=args.emb_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
        )

        # calculate latent height and width based on num_patches
        new_h, new_w = self.find_shape(num_patches)
        self.reshape = Rearrange("b (lh lw) c -> b c lh lw", lh=new_h, lw=new_w)
        self.output_shape = (args.emb_dim, new_h, new_w)

    @staticmethod
    def find_shape(num_patches: int):
        dim1 = math.ceil(math.sqrt(num_patches))
        while num_patches % dim1 != 0 and dim1 > 0:
            dim1 -= 1
        dim2 = num_patches // dim1
        return (dim1, dim2)

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        outputs = self.patch_embedding(inputs)

        outputs = self.transformer(outputs)

        outputs = outputs[:, 1:, :]  # remove CLS token

        outputs = self.reshape(outputs)

        return outputs
