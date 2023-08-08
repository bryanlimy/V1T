from .core import register, Core

import math
import torch
import typing as t
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum
from torch.utils.checkpoint import checkpoint

from v1t.models.utils import DropPath


class PatchShifting(nn.Module):
    """Patch shifting for Shifted Patch Tokenization"""

    def __init__(self, patch_size: int):
        super(PatchShifting, self).__init__()
        self.shift = int(patch_size * (1 / 2))

    def forward(self, inputs: torch.Tensor):
        """4 diagonal directions padding"""
        padded_inputs = F.pad(
            input=inputs,
            pad=(self.shift, self.shift, self.shift, self.shift),
            mode="constant",
            value=0,
        )
        left_upper = padded_inputs[..., : -self.shift * 2, : -self.shift * 2]
        right_upper = padded_inputs[..., : -self.shift * 2, self.shift * 2 :]
        left_bottom = padded_inputs[..., self.shift * 2 :, : -self.shift * 2]
        right_bottom = padded_inputs[..., self.shift * 2 :, self.shift * 2 :]
        outputs = torch.cat(
            [inputs, left_upper, right_upper, left_bottom, right_bottom],
            dim=1,
        )
        return outputs


class Image2Patches(nn.Module):
    """
    patch embedding mode:
        0 - nn.Unfold to extract patches
        1 - nn.Conv2D to extract patches
        2 - Shifted Patch Tokenization https://arxiv.org/abs/2112.13492v1
        3 - nn.Unfold with Dual PatchNorm https://openreview.net/forum?id=jgMqve6Qhw
    """

    def __init__(
        self,
        image_shape: t.Tuple[int, int, int],
        patch_mode: int,
        patch_size: int,
        stride: int,
        emb_dim: int,
        dropout: float = 0.0,
    ):
        super(Image2Patches, self).__init__()
        assert 1 <= stride <= patch_size
        c, h, w = image_shape
        self.input_shape = image_shape

        num_patches = self.unfold_dim(h, w, patch_size=patch_size, stride=stride)
        match patch_mode:
            case 0:
                patch_dim = patch_size * patch_size * c
                self.projection = nn.Sequential(
                    nn.Unfold(kernel_size=patch_size, stride=stride),
                    Rearrange("b c l -> b l c"),
                    nn.Linear(in_features=patch_dim, out_features=emb_dim),
                )
            case 1:
                self.projection = nn.Sequential(
                    nn.Conv2d(
                        in_channels=c,
                        out_channels=emb_dim,
                        kernel_size=patch_size,
                        stride=stride,
                    ),
                    Rearrange("b c h w -> b (h w) c"),
                )
            case 2:
                patch_dim = patch_size * patch_size * (c + 4)
                self.projection = nn.Sequential(
                    PatchShifting(patch_size=patch_size),
                    nn.Unfold(kernel_size=patch_size, stride=stride),
                    Rearrange("b c l -> b l c"),
                    nn.LayerNorm(normalized_shape=patch_dim),
                    nn.Linear(in_features=patch_dim, out_features=emb_dim),
                )
            case 3:
                patch_dim = patch_size * patch_size * c
                self.projection = nn.Sequential(
                    nn.Unfold(kernel_size=patch_size, stride=stride),
                    Rearrange("b c l -> b l c"),
                    nn.LayerNorm(normalized_shape=patch_dim),
                    nn.Linear(in_features=patch_dim, out_features=emb_dim),
                    nn.LayerNorm(normalized_shape=emb_dim),
                )
            case _:
                raise NotImplementedError(f"--patch_mode {patch_mode} not implemented.")
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        num_patches += 1
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, emb_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.num_patches = num_patches
        self.output_shape = (num_patches, emb_dim)

        self.apply(self.init_weight)

    @staticmethod
    def unfold_dim(h: int, w: int, patch_size: int, padding: int = 0, stride: int = 1):
        l = lambda s: math.floor(((s + 2 * padding - patch_size) / stride) + 1)
        return l(h) * l(w)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

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
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = None,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super(MLP, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.model = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)


class BehaviorMLP(nn.Module):
    def __init__(
        self,
        behavior_mode: int,
        out_dim: int,
        dropout: float = 0.0,
        mouse_ids: t.List[str] = None,
        use_bias: bool = True,
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
        in_dim = 3 if behavior_mode == 2 else 4
        mouse_ids = mouse_ids if behavior_mode == 4 else ["share"]
        self.models = nn.ModuleDict(
            {
                mouse_id: nn.Sequential(
                    nn.Linear(
                        in_features=in_dim,
                        out_features=out_dim // 2,
                        bias=use_bias,
                    ),
                    nn.Tanh(),
                    nn.Dropout(p=dropout),
                    nn.Linear(
                        in_features=out_dim // 2,
                        out_features=out_dim,
                        bias=use_bias,
                    ),
                    nn.Tanh(),
                )
                for mouse_id in mouse_ids
            }
        )

    def forward(self, inputs: torch.Tensor, mouse_id: str):
        mouse_id = mouse_id if self.behavior_mode == 4 else "share"
        return self.models[mouse_id](inputs)


class Attention(nn.Module):
    def __init__(
        self,
        num_patches: int,
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_lsa: bool = False,
        use_bias: bool = True,
        grad_checkpointing: bool = False,
    ):
        super(Attention, self).__init__()
        self.grad_checkpointing = grad_checkpointing
        inner_dim = emb_dim * num_heads

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.to_qkv = nn.Linear(
            in_features=emb_dim, out_features=inner_dim * 3, bias=False
        )
        self.rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.projection = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

        scale = emb_dim**-0.5
        if use_lsa:
            self.register_parameter(
                "scale",
                param=nn.Parameter(torch.full(size=(num_heads,), fill_value=scale)),
            )
            diagonal = torch.eye(num_patches, num_patches)
            self.register_buffer(
                "mask",
                torch.nonzero(diagonal == 1, as_tuple=False),
            )
            self.register_buffer(
                "max_value",
                torch.tensor(torch.finfo(torch.get_default_dtype()).max),
            )
        else:
            self.mask = None
            self.register_buffer("scale", torch.tensor(scale))

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        if self.mask is None:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            scale = repeat(self.scale, "h -> b h 1 1", b=q.size(0))
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -self.max_value
        attn = self.attend(dots)
        attn = self.dropout(attn)
        outputs = einsum(attn, v, "b h n i, b h i d -> b h n d")
        return outputs

    def mha(self, inputs: torch.Tensor):
        inputs = self.layer_norm(inputs)
        q, k, v = torch.chunk(self.to_qkv(inputs), chunks=3, dim=-1)
        outputs = self.scaled_dot_product_attention(
            q=self.rearrange(q), k=self.rearrange(k), v=self.rearrange(v)
        )
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.projection(outputs)
        return outputs

    def forward(self, inputs: torch.Tensor):
        if self.grad_checkpointing:
            outputs = checkpoint(
                self.mha, inputs, preserve_rng_state=True, use_reentrant=False
            )
        else:
            outputs = self.mha(inputs)
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
        mouse_ids: t.List[str],
        use_lsa: bool = False,
        drop_path: float = 0.0,
        use_bias: bool = True,
        grad_checkpointing: bool = False,
    ):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = nn.ModuleDict(
                {
                    "mha": Attention(
                        num_patches=input_shape[0],
                        emb_dim=emb_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_lsa=use_lsa,
                        use_bias=use_bias,
                        grad_checkpointing=grad_checkpointing,
                    ),
                    "mlp": MLP(
                        in_dim=emb_dim,
                        hidden_dim=mlp_dim,
                        dropout=dropout,
                        use_bias=use_bias,
                    ),
                }
            )
            if behavior_mode in (2, 3, 4):
                block["b-mlp"] = BehaviorMLP(
                    behavior_mode=behavior_mode,
                    out_dim=emb_dim,
                    mouse_ids=mouse_ids,
                    use_bias=use_bias,
                )
            self.blocks.append(block)
        self.drop_path = DropPath(dropout=drop_path)
        self.output_shape = (input_shape[0], emb_dim)

        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
    ):
        outputs = inputs
        for block in self.blocks:
            if "b-mlp" in block:
                b_latent = block["b-mlp"](behaviors, mouse_id=mouse_id)
                b_latent = repeat(b_latent, "b d -> b 1 d")
                outputs = outputs + b_latent
            outputs = self.drop_path(block["mha"](outputs)) + outputs
            outputs = self.drop_path(block["mlp"](outputs)) + outputs
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
        self.behavior_mode = args.behavior_mode

        if not hasattr(args, "grad_checkpointing"):
            args.grad_checkpointing = False
        elif args.grad_checkpointing is None:
            args.grad_checkpointing = "cuda" in args.device.type
        if args.grad_checkpointing and args.verbose:
            print(f"Enable gradient checkpointing in ViT")

        self.patch_embedding = Image2Patches(
            image_shape=input_shape,
            patch_mode=args.patch_mode,
            patch_size=args.patch_size,
            stride=args.patch_stride,
            emb_dim=args.emb_dim,
            dropout=args.p_dropout,
        )
        self.transformer = Transformer(
            input_shape=self.patch_embedding.output_shape,
            emb_dim=args.emb_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.t_dropout,
            behavior_mode=self.behavior_mode,
            mouse_ids=list(args.output_shapes.keys()),
            use_lsa=args.use_lsa,
            drop_path=args.drop_path,
            use_bias=not args.disable_bias,
            grad_checkpointing=args.grad_checkpointing,
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
        mouse_id: str,
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
