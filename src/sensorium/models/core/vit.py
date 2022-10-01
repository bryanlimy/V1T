from .core import register, Core

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
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
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
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
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
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
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

        (channels, image_height, image_width) = input_shape

        patch_size = args.patch_size
        dim = args.emb_dim
        heads = args.num_heads
        mlp_dim = args.mlp_dim
        depth = args.num_layers
        dim_head = args.dim_head
        dropout = args.dropout
        emb_dropout = args.dropout

        if isinstance(patch_size, int):
            patch_height, patch_width = patch_size, patch_size
        else:
            patch_height, patch_width = patch_size[0], patch_size[1]

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(in_features=patch_dim, out_features=dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(p=emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # calculate output_shape
        height = 32
        self._latent_dim = (height, num_patches // height, dim)
        self._output_shape = (dim, height, num_patches // height)

    def forward(self, inputs: torch.Tensor):
        outputs = self.patch_embedding(inputs)
        b, n, _ = outputs.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        outputs = torch.cat((cls_tokens, outputs), dim=1)
        outputs += self.pos_embedding[:, : n + 1]
        outputs = self.dropout(outputs)

        outputs = self.transformer(outputs)

        # remove cls_token
        outputs = outputs[:, :-1, :]

        # reshape from (num patches, patch_dim) to (HWC)
        outputs = outputs.view(*(b, *self._latent_dim))
        # reorder outputs to (CHW)
        outputs = torch.permute(outputs, dims=[0, 3, 1, 2])

        return outputs
