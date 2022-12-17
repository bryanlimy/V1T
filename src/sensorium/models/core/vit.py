from .core import register, Core

import math
import torch
import typing as t
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Image2Patches(nn.Module):
    """
    patch embedding mode:
        0 - nn.Unfold to extract overlapping patches with stride 1
        1 - nn.Unfold to extract non-overlapping patches
        2 - nn.Conv2D to extract patches with stride 1
        3 - nn.Conv2D to extract non-overlapping patches
    """

    def __init__(
        self,
        args,
        image_shape: t.Tuple[int, int, int],
        patch_size: int,
        emb_dim: int,
        dropout: float = 0.0,
    ):
        super(Image2Patches, self).__init__()
        c, h, w = image_shape
        self.input_shape = image_shape

        if not hasattr(args, "patch_mode"):
            print("patch_mode is not provided, set patch_mode to 0.")
            args.patch_mode = 0
        assert args.patch_mode in (0, 1, 2, 3)
        self.patch_mode = args.patch_mode
        stride = 1 if self.patch_mode in (0, 2) else patch_size
        num_patches = self.unfold_dim(h, w, patch_size=patch_size, stride=stride)
        patch_dim = patch_size * patch_size * c
        if self.patch_mode in (0, 1):
            self.projection = nn.Sequential(
                nn.Unfold(kernel_size=patch_size, stride=stride),
                Rearrange("b c l -> b l c"),
                nn.Linear(in_features=patch_dim, out_features=emb_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels=c,
                    out_channels=emb_dim,
                    kernel_size=patch_size,
                    stride=stride,
                ),
                Rearrange("b c h w -> b (h w) c"),
            )
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
        patches = self.projection(inputs)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        outputs = torch.cat((cls_tokens, patches), dim=1)
        outputs += self.pos_embedding
        outputs = self.dropout(outputs)
        return outputs


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int = None, dropout: float = 0.0
    ):
        super(MLP, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.model = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)


class BehaviorMLP(nn.Module):
    def __init__(
        self,
        mouse_ids: t.List[int],
        behavior_mode: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        """
        behavior mode:
        - 0: do not include behavior
        - 1: concat behavior with natural image
        - 2: add latent behavior variables to each ViT block
        - 3: add latent behavior + pupil centers to each ViT block
        - 4: separate BehaviorMLP for each animal
        """
        super(BehaviorMLP, self).__init__()
        assert behavior_mode in (2, 3, 4)
        self.behavior_mode = behavior_mode
        in_dim = 3 if behavior_mode == 2 else 5
        if behavior_mode == 4:
            self.model = nn.ModuleDict(
                {
                    str(mouse_id): nn.Sequential(
                        nn.Linear(in_features=in_dim, out_features=out_dim // 2),
                        nn.Tanh(),
                        nn.Dropout(p=dropout),
                        nn.Linear(in_features=out_dim // 2, out_features=out_dim),
                        nn.Tanh(),
                    )
                    for mouse_id in mouse_ids
                }
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim // 2),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=out_dim // 2, out_features=out_dim),
                nn.Tanh(),
            )

    def forward(self, inputs: torch.Tensor, mouse_id: int):
        if self.behavior_mode == 4:
            outputs = self.model[str(mouse_id)](inputs)
        else:
            outputs = self.model(inputs)
        return outputs


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
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, inputs: torch.Tensor):
        inputs = self.layer_norm(inputs)
        qkv = self.to_qkv(inputs).chunk(3, dim=-1)
        q, k, v = map(
            lambda a: rearrange(a, "b n (h d) -> b h n d", h=self.num_heads),
            qkv,
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
        mouse_ids: t.List[int],
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = nn.ModuleDict(
                {
                    "mha": Attention(
                        emb_dim=emb_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    ),
                    "mlp": MLP(
                        in_dim=emb_dim,
                        hidden_dim=mlp_dim,
                        dropout=dropout,
                    ),
                }
            )
            if behavior_mode > 1:
                block["b-mlp"] = BehaviorMLP(
                    mouse_ids=mouse_ids,
                    behavior_mode=behavior_mode,
                    out_dim=emb_dim,
                )
            self.blocks.append(block)
        self.output_shape = (input_shape[0], emb_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: int,
        behaviors: torch.Tensor,
    ):
        outputs = inputs
        for block in self.blocks:
            if "b-mlp" in block:
                b_latent = block["b-mlp"](behaviors, mouse_id=mouse_id)
                b_latent = repeat(b_latent, "b d -> b 1 d")
                outputs = outputs + b_latent
            outputs = block["mha"](outputs) + outputs
            outputs = block["mlp"](outputs) + outputs
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
        self.behavior_mode = args.behavior_mode
        self.patch_embedding = Image2Patches(
            args,
            image_shape=input_shape,
            patch_size=args.patch_size,
            emb_dim=emb_dim,
            dropout=args.p_dropout,
        )
        self.transformer = Transformer(
            input_shape=self.patch_embedding.output_shape,
            emb_dim=emb_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.t_dropout,
            behavior_mode=args.behavior_mode,
            mouse_ids=list(args.output_shapes.keys()),
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
        return dim1, dim2

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: int,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = self.patch_embedding(inputs)
        if self.behavior_mode in (3, 4):
            behaviors = torch.cat((behaviors, pupil_centers), dim=-1)
        outputs = self.transformer(outputs, mouse_id=mouse_id, behaviors=behaviors)
        outputs = outputs[:, 1:, :]  # remove CLS token
        outputs = self.rearrange(outputs)
        return outputs
