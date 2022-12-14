import os
import math
import torch
import argparse
import numpy as np
import typing as t
from torch import nn
import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from torch.utils.data import DataLoader


from sensorium import data
from sensorium.models.model import Model
from sensorium.utils.scheduler import Scheduler
from sensorium.utils import utils, tensorboard


from vit_visualization import Recorder, attention_rollout

utils.set_random_seed(1234)

BACKGROUND_COLOR = "#ffffff"


@torch.no_grad()
def inference(
    mouse_id: int,
    ds: DataLoader,
    model: Model,
    device: torch.device = torch.device("cpu"),
):
    latents = []
    model.to(device)
    model.train(False)
    for batch in tqdm(ds, desc=f"Mouse {mouse_id}"):
        images = batch["image"].to(device)
        behaviors = batch["behavior"].to(device)
        pupil_centers = batch["pupil_center"].to(device)
        images, _ = model.image_cropper(
            inputs=images,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        latent = model.core(
            inputs=images,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        latents.append(latent)
    latents = torch.concat(latents, dim=0)
    latents = rearrange(latents, "b n h d -> b n (h d)")
    return latents.cpu().numpy()


from sklearn.decomposition import PCA
import pandas as pd


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")
    tensorboard.set_font()

    utils.load_args(args)
    args.device = torch.device(args.device)

    _, _, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = Model(args, ds=test_ds)
    scheduler = Scheduler(args, model=model, save_optimizer=False)
    scheduler.restore(force=True)

    df = pd.DataFrame(columns=["Mouse", "Component 1", "Component 2"])
    for mouse_id, mouse_ds in test_ds.items():
        latents = inference(
            mouse_id=mouse_id, ds=mouse_ds, model=model, device=args.device
        )
        latents = np.sum(latents, axis=1)  # sum over channel dimension
        pca = PCA(n_components=2)
        pca.fit(latents)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        factors = pca.transform(latents)
        mouse_df = pd.DataFrame(
            data={
                "Mouse": np.repeat(mouse_id, repeats=len(factors)),
                "Component 1": factors[..., 0],
                "Component 2": factors[..., 1],
            }
        )
        df = pd.concat([df, mouse_df], ignore_index=True)

    import pickle

    with open("070_dim1.pkl", "wb") as file:
        pickle.dump(df, file)
        df = pickle.load(file)
    exit()

    tick_fontsize, label_fontsize = 10, 12
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=120)
    sns.scatterplot(
        data=df,
        x="Component 1",
        y="Component 2",
        hue="Mouse",
        palette="Set2",
        alpha=0.6,
        ax=ax,
    )
    x_range = np.linspace(
        np.ceil(df["Component 1"].min()), np.floor(df["Component 1"].max()), 3
    )
    tensorboard.set_xticks(
        axis=ax,
        ticks_loc=x_range,
        ticks=x_range.astype(int),
        label="Component 1",
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
    )
    y_range = np.linspace(
        np.ceil(df["Component 2"].min()), np.floor(df["Component 2"].max()), 3
    )
    tensorboard.set_yticks(
        axis=ax,
        ticks_loc=y_range,
        ticks=y_range.astype(int),
        label="Component 2",
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
    )
    sns.despine(ax=ax, offset={"left": 15, "bottom": 5}, trim=True)
    sns.move_legend(
        ax,
        loc="best",
        frameon=True,
        handletextpad=0.2,
        markerscale=0.6,
        fontsize=tick_fontsize,
        title_fontsize=tick_fontsize,
    )
    ax.set_title("Latent factors of ViT core", fontsize=label_fontsize)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")

    main(parser.parse_args())
