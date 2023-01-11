from .core import register, Core

import math
import torch
import typing as t
import numpy as np
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum
from torch.utils.checkpoint import checkpoint

from sensorium.models.utils import DropPath
from sensorium.models.core.vit import BehaviorMLP


def sinusoidal_embedding(num_channels: int, dim: int):
    pe = torch.FloatTensor(
        [
            [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
            for p in range(num_channels)
        ]
    )
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return torch.unsqueeze(pe, dim=0)


class Tokenizer(nn.Module):
    def __init__(
        self,
        image_shape: t.Tuple[int, int, int],
        patch_size: int,
        stride: int,
        emb_dim: int,
        padding: int = 3,
        dropout: float = 0.0,
        use_bias: bool = False,
        pos_emb: t.Literal["learn", "sine", "none"] = "sine",
    ):
        super(Tokenizer, self).__init__()
        assert pos_emb in ("sine", "learn", "none")

        c, h, w = image_shape
        self.image_shape = image_shape

        self.conv2d = nn.Conv2d(
            in_channels=c,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        output_shape = self.output_shape
        self.num_patches = output_shape[0]

        if pos_emb == "none":
            self.pos_embedding = None
        elif pos_emb == "learn":
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.2)
        else:
            self.register_buffer(
                "pos_embedding", sinusoidal_embedding(self.num_patches, emb_dim)
            )
        self.dropout = nn.Dropout(p=dropout)

        self.apply(self.init_weight)

    @property
    def output_shape(self):
        with torch.no_grad():
            outputs = self.tokenize(torch.zeros((1, *self.image_shape)))
        _, num_patches, emb_dim = outputs.shape
        return (num_patches, emb_dim)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def tokenize(self, inputs: torch.Tensor):
        outputs = self.conv2d(inputs)
        outputs = self.relu(outputs)
        outputs = self.max_pool2d(outputs)
        outputs = rearrange(outputs, "b c h w -> b (h w) c")
        return outputs

    def forward(self, inputs: torch.Tensor):
        outputs = self.tokenize(inputs)
        if self.pos_embedding is not None:
            outputs += self.pos_embedding
        outputs = self.dropout(outputs)
        return outputs


class Attention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        inner_dim = emb_dim // num_heads
        assert (
            inner_dim % num_heads == 0
        ), f"MHA inner_dim ({inner_dim}) must be divisible by num_heads ({num_heads})"
        # inner_dim = emb_dim * num_heads

        self.register_buffer("scale", torch.tensor(inner_dim**-0.5))
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim)

        self.qkv = nn.Linear(emb_dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(p=dropout)
        self.projection = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor):
        inputs = self.layer_norm(inputs)
        qkv = self.qkv(inputs).chunk(3, dim=-1)
        q, k, v = map(
            lambda a: rearrange(a, "b n (h d) -> b h n d", h=self.num_heads),
            qkv,
        )
        q = q * self.scale
        attn = einsum(q, k, "b h i d, b h j d -> b h i j")
        attn = self.attend(attn)
        attn = self.attn_drop(attn)
        outputs = einsum(attn, v, "b h i j, b h j d -> b h i d")
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.projection(outputs)
        return outputs


class TransformerBlock(nn.Module):
    def __init__(
        self,
        behavior_mode: int,
        emb_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        drop_path: float,
        mouse_ids: t.List[str] = None,
        grad_checkpointing: bool = False,
    ):
        super(TransformerBlock, self).__init__()
        self.grad_checkpointing = grad_checkpointing

        self.mha = Attention(emb_dim=emb_dim, num_heads=num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(normalized_shape=emb_dim),
            nn.Linear(in_features=emb_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_dim, out_features=emb_dim),
            nn.Dropout(p=dropout),
        )
        self.drop_path = DropPath(dropout=drop_path)
        self.b_mlp = None
        if behavior_mode in (1, 2, 3, 4):
            self.b_mlp = BehaviorMLP(
                behavior_mode=behavior_mode, out_dim=emb_dim, mouse_ids=mouse_ids
            )

    def checkpointing(self, fn: t.Callable, inputs: torch.Tensor):
        if self.grad_checkpointing:
            outputs = checkpoint(
                fn, inputs, preserve_rng_state=True, use_reentrant=False
            )
        else:
            outputs = fn(inputs)
        return self.drop_path(outputs) + inputs

    def forward(self, inputs: torch.Tensor, mouse_id: int, behaviors: torch.Tensor):
        outputs = inputs
        if self.b_mlp is not None:
            b_latent = self.b_mlp(behaviors, mouse_id=mouse_id)
            b_latent = repeat(b_latent, "b d -> b 1 d")
            outputs = outputs + b_latent
        outputs = self.checkpointing(self.mha, outputs)
        outputs = self.drop_path(self.mlp(outputs)) + outputs
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: t.Tuple[int, int],
        num_channels: int,
        emb_dim: int,
        mlp_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        behavior_mode: int,
        mouse_ids: t.List[str],
        drop_path: float = 0.0,
        grad_checkpointing: bool = False,
    ):
        super(Transformer, self).__init__()
        self.num_patches = input_shape[0]
        self.num_channels = num_channels

        drop_path_rates = np.linspace(0, drop_path, num_blocks)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    behavior_mode=behavior_mode,
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    drop_path=drop_path,
                    mouse_ids=mouse_ids,
                    grad_checkpointing=grad_checkpointing,
                )
                for drop_path in drop_path_rates
            ]
        )

        self.output_shape = (self.num_patches, emb_dim)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs: torch.Tensor, mouse_id: int, behaviors: torch.Tensor):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, mouse_id=mouse_id, behaviors=behaviors)
        return outputs


@register("cct")
class CCTCore(Core):
    def __init__(
        self, args, input_shape: t.Tuple[int, int, int], name: str = "CCTCore"
    ):
        super(CCTCore, self).__init__(args, input_shape=input_shape, name=name)
        self.register_buffer("reg_scale", torch.tensor(args.core_reg_scale))
        self.behavior_mode = args.behavior_mode

        if not hasattr(args, "patch_stride"):
            print("patch_stride is not defined, set to 1.")
            args.patch_stride = 1

        c, h, w = input_shape

        self.tokenizer = Tokenizer(
            image_shape=input_shape,
            patch_size=args.patch_size,
            stride=args.patch_stride,
            emb_dim=args.emb_dim,
            dropout=args.p_dropout,
            pos_emb=args.pos_emb,
        )
        tokenizer_shape = self.tokenizer.output_shape
        num_patches = tokenizer_shape[0]

        if not hasattr(args, "grad_checkpointing"):
            args.grad_checkpointing = False
        self.transformer = Transformer(
            input_shape=tokenizer_shape,
            num_channels=c,
            emb_dim=args.emb_dim,
            mlp_dim=args.mlp_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.t_dropout,
            drop_path=args.drop_path,
            behavior_mode=args.behavior_mode,
            mouse_ids=list(args.output_shapes.keys()),
            grad_checkpointing=args.grad_checkpointing,
        )

        h, w = self.find_shape(num_patches)
        self.rearrange = Rearrange("b (h w) c -> b c h w", h=h, w=w)
        self.output_shape = (self.transformer.output_shape[1], h, w)

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
        outputs = self.tokenizer(inputs)
        if self.behavior_mode in (3, 4):
            behaviors = torch.cat((behaviors, pupil_centers), dim=-1)
        outputs = self.transformer(outputs, mouse_id=mouse_id, behaviors=behaviors)
        outputs = self.rearrange(outputs)
        return outputs
