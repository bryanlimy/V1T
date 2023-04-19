import math
import torch
import numpy as np
from torch import nn
from torchvision.transforms.functional import resize

from v1t.models.core.vit import ViTCore, Attention


class Recorder(nn.Module):
    def __init__(self, core: ViTCore):
        super(Recorder, self).__init__()
        self.core = core
        self.cache = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False

    @staticmethod
    def _find_modules(module: nn.Module, type: nn.Module):
        return [m for m in module.modules() if isinstance(m, type)]

    def _hook(self, _, inputs: torch.Tensor, outputs: torch.Tensor):
        self.cache.append(outputs.clone().detach())

    def _register_hook(self):
        modules = self._find_modules(self.core.transformer, Attention)
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.core

    def clear(self):
        self.cache.clear()
        torch.cuda.empty_cache()

    def forward(
        self,
        images: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        mouse_id: str,
    ):
        """
        Return softmax scaled dot product outputs from ViT/V1T

        Returns:
            outputs: torch.Tensor, the output of the core
            attentions: torch.Tensor, the softmax scaled dot product outputs in
                format (batch size, num blocks, num heads, num patches, num patches)
        """
        assert not self.ejected, "recorder has been ejected, cannot be used anymore"
        self.clear()
        if not self.hook_registered:
            self._register_hook()
        outputs = self.core(
            inputs=images,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            mouse_id=mouse_id,
        )
        attentions = None
        if len(self.cache) > 0:
            attentions = torch.stack(self.cache, dim=1)
        return outputs, attentions


def find_shape(num_patches: int):
    dim1 = math.ceil(math.sqrt(num_patches))
    while num_patches % dim1 != 0 and dim1 > 0:
        dim1 -= 1
    dim2 = num_patches // dim1
    return dim1, dim2


import typing as t


def normalize(x: t.Union[np.ndarray, torch.Tensor]):
    return (x - x.min()) / (x.max() - x.min())


def attention_rollout(attention: torch.Tensor, image_shape: t.List[int, int]):
    """
    Apply Attention rollout from https://arxiv.org/abs/2005.00928 to a single
    sample of softmax attention
    Code examples
    - https://keras.io/examples/vision/probing_vits/#method-ii-attention-rollout
    - https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    """
    assert attention.dim() == 4
    with torch.device(attention.device):
        # take max values of attention heads
        attention, _ = torch.max(attention, dim=1)

        # to account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(attention.size(1))
        aug_att_mat = attention + residual_att
        aug_att_mat = aug_att_mat / torch.sum(aug_att_mat, dim=-1, keepdim=True)

        # recursively multiply the weight matrices
        joint_attentions = torch.zeros_like(aug_att_mat)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        heatmap = joint_attentions[-1, 0, 1:]
        heatmap = torch.reshape(heatmap, shape=find_shape(len(heatmap)))
        heatmap = normalize(heatmap)
        heatmap = resize(heatmap[None, ...], size=image_shape, antialias=False)
    return heatmap[0]


def attention_rollouts(attentions: torch.Tensor, image_shape: t.List[int, int]):
    """Apply attention rollout to a batch of softmax attentions"""
    assert attentions.dim() == 5
    batch_size = attentions.size(0)
    with torch.device(attentions.device):
        heatmaps = torch.zeros((batch_size, *image_shape))
        for i in range(batch_size):
            heatmaps[i] = attention_rollout(attentions[i], image_shape=image_shape)
    return heatmaps
