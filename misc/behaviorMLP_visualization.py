import os
import math
import joypy
import torch
import argparse
import numpy as np
import typing as t
import pandas as pd
from torch import nn
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sensorium import data
from sensorium.models.model import Model
from sensorium.utils.scheduler import Scheduler
from sensorium.utils import utils, tensorboard
from sensorium.models.core.vit import ViTCore, BehaviorMLP


utils.set_random_seed(1234)

BACKGROUND_COLOR = "#ffffff"


def convert(mouse_id: str):
    pairs = {"2": "A", "3": "B", "4": "C", "5": "D", "6": "E"}
    return pairs[mouse_id] if mouse_id in pairs else mouse_id


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

    def _register_hook(self, mouse_id: str):
        modules = self._find_modules(self.vit.transformer, BehaviorMLP)
        for module in modules:
            handle = module.model.register_forward_hook(self._hook)
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
    results: np.ndarray,
    mouse_id: str,
    filename: str = None,
    colormap: str = "Set2",
):
    df = pd.DataFrame()
    for i in range(len(results)):
        df[i + 1] = results[i]

    tick_fontsize, label_fontsize = 8, 10
    figure, axes = joypy.joyplot(
        df,
        figsize=(4, 4),
        colormap=cm.get_cmap(colormap),
        alpha=0.8,
        overlap=1.5,
        # kind="normalized_counts",
        bins=30,
        range_style="all",
        x_range=[-1.25, 1.25],
        linewidth=1.5,
        title=f"Mouse {convert(mouse_id)} B-MLP activations",
    )
    pos = axes[0].get_position()
    axes[0].text(
        x=-1.4,
        y=pos.y1 + 0.5,
        s="Block",
        fontsize=label_fontsize,
        ha="left",
        va="center",
    )
    axes[-1].set_xlabel("Average activation distribution", fontsize=label_fontsize)
    tensorboard.set_ticks_params(axis=axes[-1])
    axes[-1].xaxis.set_tick_params(length=4, pad=3, width=1)

    plt.show()
    if filename is not None:
        # tensorboard.save_figure(figure, filename=filename, dpi=120)
        print(f"plot saved to {filename}.")


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    tensorboard.set_font()

    utils.load_args(args)
    args.batch_size = 1
    args.device = torch.device(args.device)

    _, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.data,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = Model(args, ds=val_ds)
    model.train(False)

    scheduler = Scheduler(args, model=model, save_optimizer=False)
    scheduler.restore(force=True)

    results = {}
    for mouse_id, mouse_ds in test_ds.items():
        if mouse_id == "1":
            continue
        recorder = Recorder(model.core)
        result = []
        for batch in tqdm(mouse_ds, desc=f"Mouse {mouse_id}"):
            with torch.no_grad():
                behavior = batch["behavior"]
                pupil_center = batch["pupil_center"]
                image, _ = model.image_cropper(
                    inputs=batch["image"],
                    mouse_id=mouse_id,
                    behaviors=behavior,
                    pupil_centers=pupil_center,
                )
                activations = recorder(
                    images=image,
                    behaviors=behavior,
                    pupil_centers=pupil_center,
                    mouse_id=mouse_id,
                )
                recorder.clear()
            result.append(activations.numpy())
        result = np.array(result)
        # compute the average activation for every block over all samples
        result = np.mean(result, axis=0)
        results[mouse_id] = result
        recorder.eject()
        del recorder

    import pickle

    with open(os.path.join(args.output_dir, "behaviorMLP.pkl"), "wb") as file:
        # results = pickle.load(file)
        pickle.dump(results, file)
    exit()

    plot_distribution_map(
        results=results,
        mouse_id=mouse_id,
        filename=os.path.join(
            args.output_dir,
            f"behaviorMLP_mouse{mouse_id}.svg",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    main(parser.parse_args())
