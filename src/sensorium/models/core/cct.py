from .core import register, Core

import math
import torch
import typing as t
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum

from sensorium.models.utils import DropPath
from sensorium.models.core.vit import BehaviorMLP


class Tokenizer(nn.Module):
    def __init__(
        self,
        image_shape: t.Tuple[int, int, int],
        patch_size: int,
        stride: int,
        emb_dim: int,
        padding: int = 3,
        use_bias: bool = False,
        num_layers: int = 1,
    ):
        super(Tokenizer, self).__init__()
        c, h, w = image_shape
        self.image_shape = image_shape

        n_filter_list = [c] + [emb_dim for _ in range(num_layers - 1)] + [emb_dim]
        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        layers = []
        for in_channels, out_channels in n_filter_list_pairs:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=patch_size,
                        stride=stride,
                        padding=padding,
                        bias=use_bias,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=6, stride=1, padding=1),
                ]
            )
        self.tokenizer = nn.Sequential(*layers)
        self.apply(self.init_weight)

    @property
    def output_shape(self):
        with torch.no_grad():
            inputs = torch.zeros((1, *self.image_shape))
            outputs = self.forward(inputs)
            _, num_patches, emb_dim = outputs.shape
        return (num_patches, emb_dim)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs: torch.Tensor):
        outputs = self.tokenizer(inputs)
        outputs = rearrange(outputs, "b c h w -> b (h w) c")
        return outputs


class Attention(nn.Module):
    def __init__(
        self,
        num_patches: int,
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = emb_dim // num_heads
        inner_dim = emb_dim * num_heads

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
        num_patches: int,
        emb_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        drop_path: float,
        mouse_ids: t.List[int] = None,
    ):
        super(TransformerBlock, self).__init__()
        self.mha = Attention(
            num_patches=num_patches,
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
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

    def forward(self, inputs: torch.Tensor, mouse_id: int, behaviors: torch.Tensor):
        outputs = inputs
        if self.b_mlp is not None:
            b_latent = self.b_mlp(behaviors, mouse_id=mouse_id)
            b_latent = repeat(b_latent, "b d -> b 1 d")
            outputs = outputs + b_latent
        outputs = self.drop_path(self.mha(outputs)) + outputs
        outputs = self.drop_path(self.mlp(outputs)) + outputs
        return outputs


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


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: t.Tuple[int, int],
        num_channels: int,
        emb_dim: int,
        num_blocks: int,
        num_heads: int,
        p_dropout: float,
        t_dropout: float,
        behavior_mode: int,
        mouse_ids: t.List[int],
        drop_path: float = 0.0,
        mlp_ratio: float = 3,
        pos_emb: t.Literal["sine", "learn", "none"] = "sine",
    ):
        super(Transformer, self).__init__()
        assert pos_emb in ("sine", "learn", "none")

        self.num_patches = input_shape[0]
        self.num_channels = num_channels

        if pos_emb == "none":
            self.pos_emb = None
        elif pos_emb == "learn":
            self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.2)
        else:
            self.register_buffer(
                "pos_emb", sinusoidal_embedding(self.num_patches, emb_dim)
            )

        self.p_dropout = nn.Dropout(p=p_dropout)

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]

        mlp_dim = int(emb_dim * mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    behavior_mode=behavior_mode,
                    num_patches=self.num_patches,
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=t_dropout,
                    drop_path=drop_path_rates[i],
                    mouse_ids=mouse_ids,
                )
                for i in range(len(drop_path_rates))
            ]
        )

        self.output_shape = (self.num_patches, emb_dim)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs: torch.Tensor, mouse_id: int, behaviors: torch.Tensor):
        outputs = inputs
        if self.pos_emb is None and inputs.size(1) < self.num_patches:
            outputs = F.pad(
                outputs,
                (0, 0, 0, self.num_channels - outputs.size(1)),
                mode="constant",
                value=0,
            )
        if self.pos_emb is not None:
            outputs += self.pos_emb
        outputs = self.p_dropout(outputs)
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
            num_layers=1,
        )
        tokenizer_shape = self.tokenizer.output_shape
        num_patches = tokenizer_shape[0]

        self.transformer = Transformer(
            input_shape=tokenizer_shape,
            num_channels=c,
            emb_dim=args.emb_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            p_dropout=args.p_dropout,
            t_dropout=args.t_dropout,
            drop_path=args.drop_path,
            mlp_ratio=args.mlp_ratio,
            behavior_mode=args.behavior_mode,
            mouse_ids=list(args.output_shapes.keys()),
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
