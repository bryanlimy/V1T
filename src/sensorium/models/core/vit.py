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
        for _ in range(num_layers):
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


class Image2Patches(nn.Module):
    def __init__(
        self,
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
        self.num_patches = new_h * new_w

    def forward(self, inputs: torch.Tensor):
        patches = inputs.unfold(
            dimension=2, size=self.patch_size, step=self.stride
        ).unfold(dimension=3, size=self.patch_size, step=self.stride)
        patches = self.rearrange(patches)
        return patches


def find_shape(num_patches: int):
    num1 = math.floor(math.sqrt(num_patches))
    while num_patches % num1 != 0:
        num1 -= 1
    num2 = num_patches // num1
    return num1, num2


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
        emb_dim = args.emb_dim
        heads = args.num_heads
        mlp_dim = args.mlp_dim
        num_layers = args.num_layers
        dim_head = args.dim_head
        dropout = args.dropout
        emb_dropout = args.dropout

        patch_dim = patch_size * patch_size * c
        self.image2patches = Image2Patches(
            image_shape=input_shape,
            patch_size=patch_size,
            stride=1 if args.crop_mode else patch_size,
        )
        num_patches = self.image2patches.num_patches
        self.patches2emb = nn.Linear(in_features=patch_dim, out_features=emb_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        num_patches += 1  # cls token

        self.transformer = Transformer(
            dim=emb_dim,
            num_layers=num_layers,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # calculate latent height and width based on num_patches
        (latent_height, latent_width) = find_shape(num_patches)
        self.output_shape = (emb_dim, latent_height, latent_width)

        # reshape transformer output from (batch_size, num_patches, channels)
        # to (batch_size, channel, latent height, latent width)
        self.output_layer = Rearrange(
            "b (lh lw) c -> b c lh lw",
            lh=latent_height,
            lw=latent_width,
            c=emb_dim,
        )

    def forward(self, inputs: torch.Tensor):
        outputs = self.image2patches(inputs)
        outputs = self.patches2emb(outputs)

        batch_size, num_patches, _ = outputs.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        outputs = torch.cat((cls_tokens, outputs), dim=1)
        outputs += self.pos_embedding[:, : num_patches + 1]
        outputs = self.emb_dropout(outputs)

        outputs = self.transformer(outputs)

        outputs = self.output_layer(outputs)

        return outputs
