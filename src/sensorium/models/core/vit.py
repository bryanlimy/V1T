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
        self.input_shape = image_shape

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        num_patches = self.unfold_dim(h, w, patch_size=patch_size, stride=stride)
        patch_dim = patch_size * patch_size * c

        self.rearrange = Rearrange("b c l -> b l c")
        self.linear = nn.Linear(in_features=patch_dim, out_features=emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        num_patches += 1
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, emb_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.num_patches = num_patches
        self.output_shape = (num_patches, emb_dim)

    @staticmethod
    def unfold_dim(h: int, w: int, patch_size: int, padding: int = 0, stride: int = 1):
        l = lambda s: math.floor(((s + 2 * padding - patch_size) / stride) + 1)
        return l(h) * l(w)

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.size(0)
        patches = self.unfold(inputs)
        patches = self.rearrange(patches)
        outputs = self.linear(patches)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        outputs = torch.cat((cls_tokens, outputs), dim=1)
        outputs += self.pos_embedding
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
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int = None, dropout: float = 0.0
    ):
        super(FeedForward, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
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
        input_shape: t.Tuple[int, int],
        emb_dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        behavior_mode: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            emb_dim += 3
            block = nn.ModuleDict(
                {
                    "attn": PreNorm(
                        dim=emb_dim,
                        fn=Attention(
                            emb_dim=emb_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                        ),
                    ),
                    "ff": PreNorm(
                        dim=emb_dim,
                        fn=FeedForward(
                            in_dim=emb_dim,
                            hidden_dim=mlp_dim,
                            dropout=dropout,
                        ),
                    ),
                }
            )
            # if behavior_mode == 2:
            #     block.update(
            #         {
            #             "bff": nn.Sequential(
            #                 nn.Linear(in_features=3, out_features=emb_dim // 2),
            #                 nn.Tanh(),
            #                 nn.Linear(in_features=emb_dim // 2, out_features=emb_dim),
            #             )
            #         }
            #     )
            self.blocks.append(block)
        self.output_shape = (input_shape[0], emb_dim)

    def forward(self, inputs: torch.Tensor, behaviors: torch.Tensor):
        outputs = inputs
        behaviors = repeat(behaviors, "b d -> b l d", l=outputs.size(1))
        for block in self.blocks:
            # if "bff" in block:
            #     b_latent = block["bff"](behaviors)
            #     b_latent = repeat(b_latent, "b d -> b 1 d")
            #     outputs = outputs + b_latent
            outputs = torch.cat((outputs, behaviors), dim=-1)
            outputs = block["attn"](outputs) + outputs
            outputs = block["ff"](outputs) + outputs
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
        self.register_buffer("reg_scale", torch.tensor(args.core_reg_scale))
        emb_dim = args.emb_dim
        self.patch_embedding = Image2Patches(
            image_shape=input_shape,
            patch_size=args.patch_size,
            stride=1,
            emb_dim=emb_dim,
            dropout=args.dropout,
        )
        self.transformer = Transformer(
            input_shape=self.patch_embedding.output_shape,
            emb_dim=emb_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            behavior_mode=args.behavior_mode,
        )

        # calculate latent height and width based on num_patches
        h, w = self.find_shape(self.patch_embedding.num_patches - 1)
        self.rearrange = Rearrange("b (h w) c -> b c h w", h=h, w=w)
        self.output_shape = (self.transformer.output_shape[-1], h, w)

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

    def forward(self, inputs: torch.Tensor, behaviors: torch.Tensor):
        outputs = self.patch_embedding(inputs)
        outputs = self.transformer(outputs, behaviors=behaviors)
        outputs = outputs[:, 1:, :]  # remove CLS token
        outputs = self.rearrange(outputs)
        return outputs
