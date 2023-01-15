import os
import math
import torch
import argparse
import numpy as np
import typing as t
from torch import nn
import matplotlib.cm as cm
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize

from sensorium import data
from sensorium.models.model import Model
from sensorium.utils.scheduler import Scheduler
from sensorium.utils import utils, tensorboard
from sensorium.models.core.vit import ViTCore, Attention


utils.set_random_seed(1234)

BACKGROUND_COLOR = "#ffffff"

MOUSE_ID = "2"
# MOUSE_ID = "static26085-6-3"


def find_shape(num_patches: int):
    dim1 = math.ceil(math.sqrt(num_patches))
    while num_patches % dim1 != 0 and dim1 > 0:
        dim1 -= 1
    dim2 = num_patches // dim1
    return dim1, dim2


def normalize(x: np.ndarray):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


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


def to_rgb(image: np.ndarray):
    assert image.shape[0] == 2
    blue, green = image[0][..., None], image[1][..., None]
    red = np.zeros_like(blue)
    return np.concatenate((red, blue, green), axis=-1)


def plot_attention_map(
    results: t.List[t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    filename: str = None,
    colormap: str = "turbo",
    alpha: float = 0.5,
):
    cmap = cm.get_cmap(colormap)
    colors = cmap(np.arange(256))[:, :3]
    label_fontsize, tick_fontsize = 10, 8
    figure, axes = plt.subplots(
        nrows=len(results),
        ncols=2,
        figsize=(7, 2.2 * len(results)),
        gridspec_kw={"wspace": -0.25, "hspace": 0.2},
        dpi=120,
        facecolor=BACKGROUND_COLOR,
    )
    for i, (image, heatmap, behavior, pupil_center) in enumerate(results):
        gray_image = image.shape[0] == 1
        image = image[0] if gray_image else to_rgb(image)
        axes[i, 0].imshow(image.astype(np.uint8), cmap="gray")
        heatmap = colors[np.uint8(255.0 * heatmap)] * 255.0
        image = image[..., None] if gray_image else image
        heatmap = alpha * heatmap + (1 - alpha) * image
        # heatmap = heatmap * image
        axes[i, 1].imshow(heatmap.astype(np.uint8), cmap=colormap)
        if i == 0:
            axes[i, 0].set_title("Input", fontsize=label_fontsize)
            axes[i, 1].set_title("Attention rollout", fontsize=label_fontsize)
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

        axes[i, 0].text(
            x=0.3,
            y=-0.08,
            s=f"pupil dilation {behavior[0]:.02f}, "
            f"derivative: {behavior[1]:.02f}, "
            f"speed: {behavior[2]:.02f}, "
            f"pupil center: ({pupil_center[0]:.02f}, {pupil_center[1]:.02f})",
            fontsize=tick_fontsize,
            ha="left",
            transform=axes[i, 0].transAxes,
        )

    # plot colorbar
    pos1 = axes[0, 1].get_position()
    pos2 = axes[-1, 1].get_position()
    width, height = 0.01, (pos1.y1 - pos1.y0) * 0.35
    cbar_ax = figure.add_axes(
        rect=[
            pos1.x1 + 0.01,
            ((pos1.y1 - pos2.y0) / 2 + pos2.y0) - (height / 2),
            width,
            height,
        ]
    )
    figure.colorbar(cm.ScalarMappable(cmap=colormap), cax=cbar_ax, shrink=0.1)
    tensorboard.set_yticks(
        axis=cbar_ax,
        ticks_loc=np.linspace(0, 1, 3),
        tick_fontsize=tick_fontsize,
    )
    tensorboard.set_ticks_params(axis=cbar_ax)
    plt.show()
    if filename is not None:
        # tensorboard.save_figure(figure, filename=filename, dpi=120)
        print(f"plot saved to {filename}.")


def plot_attention_map_2(
    val_results: t.List[t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    test_results: t.List[t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    filename: str = None,
    colormap: str = "turbo",
    alpha: float = 0.5,
):
    assert len(val_results) == len(test_results) == 3
    cmap = cm.get_cmap(colormap)
    colors = cmap(np.arange(256))[:, :3]
    label_fontsize, tick_fontsize = 10, 8
    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(8, 4),
        gridspec_kw={"wspace": 0.05, "hspace": -0.25},
        dpi=240,
        facecolor=BACKGROUND_COLOR,
    )
    for i, (image, heatmap, behavior, pupil_center) in enumerate(val_results):
        gray_image = image.shape[0] == 1
        image = image[0] if gray_image else to_rgb(image)
        heatmap = colors[np.uint8(255.0 * heatmap)] * 255.0
        image = image[..., None] if gray_image else image
        heatmap = alpha * heatmap + (1 - alpha) * image
        # heatmap = heatmap * image
        axes[0, i].imshow(heatmap.astype(np.uint8), cmap=colormap)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        tensorboard.remove_spines(axis=axes[0, i])
        description = (
            f"[{behavior[0]:.01f}, "  # pupil dilation
            f"{behavior[1]:.01f}, "  # dilation derivative
            f"({pupil_center[0]:.01f}, {pupil_center[1]:.01f}), "  # pupil center
            f"{behavior[2]:.01f}]"  # speed
        )
        axes[0, i].set_xlabel(description, labelpad=0, fontsize=tick_fontsize)
    axes[0, 0].set_ylabel("Validation samples", labelpad=0.05, fontsize=tick_fontsize)
    for i, (image, heatmap, behavior, pupil_center) in enumerate(test_results):
        gray_image = image.shape[0] == 1
        image = image[0] if gray_image else to_rgb(image)
        heatmap = colors[np.uint8(255.0 * heatmap)] * 255.0
        image = image[..., None] if gray_image else image
        heatmap = alpha * heatmap + (1 - alpha) * image
        # heatmap = heatmap * image
        axes[1, i].imshow(heatmap.astype(np.uint8), cmap=colormap)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        tensorboard.remove_spines(axis=axes[1, i])
        description = (
            f"[{behavior[0]:.01f}, "  # pupil dilation
            f"{behavior[1]:.01f}, "  # dilation derivative
            f"({pupil_center[0]:.01f}, {pupil_center[1]:.01f}), "  # pupil center
            f"{behavior[2]:.01f}]"  # speed
        )
        axes[1, i].set_xlabel(description, labelpad=0, fontsize=tick_fontsize)
    axes[-1, 0].set_ylabel("Test samples", labelpad=0.05, fontsize=tick_fontsize)
    # figure.suptitle(
    #     r"$\bf{A}$ Attention rollout visualization",
    #     y=axes[0].get_position().y1 + 0.05,
    #     fontsize=tick_fontsize,
    # )

    # plot colorbar
    pos1 = axes[0, -1].get_position()
    pos2 = axes[-1, -1].get_position()
    width, height = 0.008, (pos1.y1 - pos1.y0) * 0.35
    cbar_ax = figure.add_axes(
        rect=[
            pos1.x1 + 0.01,
            ((pos1.y1 - pos2.y0) / 2 + pos2.y0) - (height / 2),
            width,
            height,
        ]
    )
    figure.colorbar(cm.ScalarMappable(cmap=colormap), cax=cbar_ax, shrink=0.1)
    tensorboard.set_yticks(
        axis=cbar_ax,
        ticks_loc=np.linspace(0, 1, 3),
        tick_fontsize=tick_fontsize,
    )
    tensorboard.set_ticks_params(axis=cbar_ax)

    plt.show()
    if filename is not None:
        tensorboard.save_figure(figure, filename=filename, dpi=120)
        print(f"plot saved to {filename}.")


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
    # heatmap = heatmap / np.max(heatmap)
    heatmap = normalize(heatmap)
    heatmap = resize(
        heatmap,
        output_shape=image.shape[1:],
        preserve_range=True,
        anti_aliasing=False,
    )
    return heatmap


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")
    tensorboard.set_font()

    utils.load_args(args)
    args.batch_size = 1
    args.device = torch.device(args.device)

    _, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = Model(args, ds=val_ds)
    model.train(False)

    scheduler = Scheduler(args, model=model, save_optimizer=False)
    scheduler.restore(force=True)

    num_plots = 3
    recorder = Recorder(model.core)

    test_results = []
    for batch in test_ds[MOUSE_ID]:
        with torch.no_grad():
            pupil_center = batch["pupil_center"]
            # pupil_center = torch.zeros_like(pupil_center)
            behavior = batch["behavior"]
            # behavior = torch.zeros_like(behavior)
            image, _ = model.image_cropper(
                inputs=batch["image"],
                mouse_id=MOUSE_ID,
                behaviors=behavior,
                pupil_centers=pupil_center,
            )
            _, attention = recorder(
                images=image,
                behaviors=behavior,
                pupil_centers=pupil_center,
                mouse_id=MOUSE_ID,
            )
            image = val_ds[MOUSE_ID].dataset.i_transform_image(image)
            recorder.clear()
        image, attention = image.numpy()[0], attention.numpy()[0]
        heatmap = attention_rollout(image=image, attention=attention)
        test_results.append(
            (image, heatmap, behavior.numpy()[0], pupil_center.numpy()[0])
        )
        if len(test_results) == num_plots:
            break

    val_results = []
    for batch in val_ds[MOUSE_ID]:
        with torch.no_grad():
            pupil_center = batch["pupil_center"]
            # pupil_center = torch.zeros_like(pupil_center)
            behavior = batch["behavior"]
            # behavior = torch.zeros_like(behavior)
            image, _ = model.image_cropper(
                inputs=batch["image"],
                mouse_id=MOUSE_ID,
                behaviors=behavior,
                pupil_centers=pupil_center,
            )
            _, attention = recorder(
                images=image,
                behaviors=behavior,
                pupil_centers=pupil_center,
                mouse_id=MOUSE_ID,
            )
            image = val_ds[MOUSE_ID].dataset.i_transform_image(image)
            recorder.clear()
        image, attention = image.numpy()[0], attention.numpy()[0]
        heatmap = attention_rollout(image=image, attention=attention)
        val_results.append(
            (image, heatmap, behavior.numpy()[0], pupil_center.numpy()[0])
        )
        if len(val_results) == num_plots:
            break

    plot_attention_map_2(
        val_results=val_results,
        test_results=test_results,
        filename=os.path.join(
            args.output_dir, "plots", f"attention_rollout_mouse{MOUSE_ID}.svg"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    main(parser.parse_args())
