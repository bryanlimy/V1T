from .core import register, Core

import math
import torch
import typing as t
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

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
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim=dim,
                            fn=Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim=dim, fn=FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        for attn, ff in self.layers:
            outputs = attn(outputs) + outputs
            outputs = ff(outputs) + outputs
        return outputs


class Image2Patches(nn.Module):
    def __init__(
        self,
        model,
        image_shape: t.Tuple[int, int, int],
        patch_size: int,
        stride: int = 1,
    ):
        super(Image2Patches, self).__init__()
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
        model.num_patches = new_h * new_w

    def forward(self, inputs: torch.Tensor):
        patches = inputs.unfold(
            dimension=2, size=self.patch_size, step=self.stride
        ).unfold(dimension=3, size=self.patch_size, step=self.stride)
        patches = self.rearrange(patches)
        return patches


def find_shape(num_patches: int):
    dim1 = math.ceil(math.sqrt(num_patches))
    while num_patches % dim1 != 0 and dim1 > 0:
        dim1 -= 1
    dim2 = num_patches // dim1
    return (dim1, dim2)


@register("vit")
class ViTCore(Core):
    def __init__(
        self,
        args,
        input_shape: t.Tuple[int, int, int],
        name: str = "ViTCore",
    ):
        super(ViTCore, self).__init__(args, input_shape=input_shape, name=name)
        (c, h, w) = input_shape
        patch_size = args.patch_size
        patch_stride = 1 if max(input_shape) < 100 else patch_size
        emb_dim = args.emb_dim
        heads = args.num_heads
        mlp_dim = args.mlp_dim
        num_layers = args.num_layers
        dim_head = args.dim_head
        dropout = args.dropout
        emb_dropout = args.dropout
        self.reg_scale = torch.tensor(args.core_reg_scale, device=args.device)

        if isinstance(patch_size, int):
            patch_height, patch_width = patch_size, patch_size
        else:
            patch_height, patch_width = patch_size[0], patch_size[1]

        patch_dim = patch_height * patch_width * c
        self.patch_embedding = nn.Sequential(
            Image2Patches(
                self,
                image_shape=input_shape,
                patch_size=patch_size,
                stride=patch_stride,
            ),
            nn.Linear(in_features=patch_dim, out_features=emb_dim),
        )
        num_patches = self.num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.transformer = Transformer(
            dim=emb_dim,
            num_layers=num_layers,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # calculate latent height and width based on num_patches
        (new_h, new_w) = find_shape(num_patches)
        self.latent_dim = (new_h, new_w, emb_dim)
        self.output_shape = (emb_dim, new_h, new_w)

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        outputs = self.patch_embedding(inputs)
        b, n, _ = outputs.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        outputs = torch.cat((cls_tokens, outputs), dim=1)
        outputs += self.pos_embedding[:, : n + 1]
        outputs = self.emb_dropout(outputs)

        outputs = self.transformer(outputs)

        # remove cls_token
        outputs = outputs[:, :-1, :]

        # reshape from (num patches, patch_dim) to (HWC)
        outputs = outputs.view(*(b, *self.latent_dim))
        # reorder outputs to (CHW)
        outputs = torch.permute(outputs, dims=[0, 3, 1, 2])

        return outputs
