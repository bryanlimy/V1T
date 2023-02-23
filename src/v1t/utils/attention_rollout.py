import math
import torch
import numpy as np
from torch import nn
from skimage.transform import resize

from v1t.models.core.vit import ViTCore, Attention


class Recorder(nn.Module):
    def __init__(self, vit: ViTCore, device: str = "cpu"):
        super().__init__()
        self.vit = vit

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    @staticmethod
    def _find_modules(nn_module, type):
        return [module for module in nn_module.modules() if isinstance(module, type)]

    def _register_hook(self):
        modules = self._find_modules(self.vit.transformer, Attention)
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(
        self,
        images: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        mouse_id: str,
    ):
        """Return attention output from ViT
        attns has shape (batch size, num blocks, num heads, num patches, num patches)
        """
        assert not self.ejected, "recorder has been ejected, cannot be used anymore"
        self.clear()
        if not self.hook_registered:
            self._register_hook()
        pred = self.vit(
            inputs=images,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            mouse_id=mouse_id,
        )
        recordings = tuple(map(lambda tensor: tensor.to(self.device), self.recordings))
        attns = torch.stack(recordings, dim=1) if len(recordings) > 0 else None
        return pred, attns


def find_shape(num_patches: int):
    dim1 = math.ceil(math.sqrt(num_patches))
    while num_patches % dim1 != 0 and dim1 > 0:
        dim1 -= 1
    dim2 = num_patches // dim1
    return dim1, dim2


def normalize(x: np.ndarray):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def attention_rollout(image: np.ndarray, attention: np.ndarray):
    """
    Attention rollout from https://arxiv.org/abs/2005.00928
    Code examples
    - https://keras.io/examples/vision/probing_vits/#method-ii-attention-rollout
    - https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    """
    # average the attention heads
    attention = np.max(attention, axis=1)

    # to account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = np.eye(attention.shape[1])
    aug_att_mat = attention + residual_att
    aug_att_mat = aug_att_mat / np.expand_dims(aug_att_mat.sum(axis=-1), axis=-1)

    # recursively multiply the weight matrices
    joint_attentions = np.zeros(aug_att_mat.shape)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.shape[0]):
        joint_attentions[n] = np.matmul(aug_att_mat[n], joint_attentions[n - 1])

    heatmap = joint_attentions[-1, 0, 1:]
    heatmap = np.reshape(heatmap, newshape=find_shape(len(heatmap)))
    heatmap = normalize(heatmap)
    heatmap = resize(
        heatmap,
        output_shape=image.shape[1:],
        preserve_range=True,
        anti_aliasing=False,
    )
    return heatmap
