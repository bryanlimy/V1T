import os
import math
import joypy
import torch
import argparse
import numpy as np
import typing as t
import pandas as pd
from torch import nn
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from sensorium import data
from sensorium.models.model import Model
from sensorium.utils.scheduler import Scheduler
from sensorium.utils import utils, tensorboard
from sensorium.models.core.vit import ViTCore, BehaviorMLP


utils.set_random_seed(1234)

BACKGROUND_COLOR = "#ffffff"

font_path = "/Users/bryanlimy/Git/Lexend/Lexend-Regular.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()


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

    def _hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.recordings.append(output.clone().detach())

    @staticmethod
    def _find_modules(nn_module, type):
        return [module for module in nn_module.modules() if isinstance(module, type)]

    def _register_hook(self, mouse_id: int):
        modules = self._find_modules(self.vit.transformer, BehaviorMLP)
        for module in modules:
            handle = module.models[str(mouse_id)].register_forward_hook(self._hook)
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
        mouse_id: int,
    ):
        """Return attention output from ViT
        attns has shape (batch size, num blocks, num heads, num patches, num patches)
        """
        assert not self.ejected, "recorder has been ejected, cannot be used anymore"
        self.clear()
        if not self.hook_registered:
            self._register_hook(mouse_id=mouse_id)
        _ = self.vit(
            inputs=images,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            mouse_id=mouse_id,
        )
        activations = torch.vstack(self.recordings)
        return activations


def plot_distribution_map(
    results: np.ndarray, mouse_id: int, filename: str = None, colormap: str = "Set2"
):
    df = pd.DataFrame()
    for i in range(len(results)):
        df[i + 1] = results[i]

    tick_fontsize, label_fontsize = 8, 10
    figure, axes = joypy.joyplot(
        df,
        figsize=(5, 6),
        colormap=cm.get_cmap(colormap),
        alpha=0.8,
        overlap=1.25,
        # kind="normalized_counts",
        bins=30,
        range_style="all",
        x_range=[-1.5, 1.5],
        linewidth=1.5,
        title=f"Mouse {mouse_id} behavior activations per layer",
    )
    pos = axes[0].get_position()
    axes[0].text(
        x=-1.65,
        y=pos.y1 - 0.5,
        s="Layer",
        fontsize=label_fontsize,
        ha="left",
        va="center",
    )
    axes[-1].set_xlabel("Average activation distribution", fontsize=label_fontsize)
    tensorboard.set_ticks_params(axis=axes[-1])
    axes[-1].xaxis.set_tick_params(length=4, pad=3, width=1)

    plt.show()
    if filename is not None:
        tensorboard.save_figure(figure, filename=filename, dpi=120)
        print(f"plot saved to {filename}.")


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

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

    for mouse_id, mouse_ds in val_ds.items():
        recorder = Recorder(model.core)
        results = []
        for batch in val_ds[mouse_id]:
            with torch.no_grad():
                pupil_center = batch["pupil_center"]
                behavior = batch["behavior"]
                image, _ = model.image_cropper(
                    inputs=batch["image"],
                    mouse_id=mouse_id,
                    pupil_centers=pupil_center,
                    behaviors=behavior,
                )
                activations = recorder(
                    images=image,
                    behaviors=behavior,
                    pupil_centers=pupil_center,
                    mouse_id=mouse_id,
                )
                recorder.clear()
            results.append(activations.numpy())
        results = np.array(results)
        # compute the average activation for every block over all samples
        results = np.mean(results, axis=0)
        plot_distribution_map(
            results=results,
            mouse_id=mouse_id,
            filename=os.path.join(
                "plots", "b-mlp_distributions", f"mouse{mouse_id}.jpg"
            ),
        )
        del recorder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    main(parser.parse_args())
