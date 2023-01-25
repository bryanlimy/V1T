import os
import torch
import argparse
import numpy as np
import typing as t
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from sensorium import data
from sensorium.models.model import Model
from sensorium.losses import correlation
from sensorium.utils import utils, tensorboard
from sensorium.utils.scheduler import Scheduler


sns.set_style("ticks")
utils.set_random_seed(1234)

BACKGROUND_COLOR = "#ffffff"


@torch.no_grad()
def inference(
    model: Model, ds: DataLoader, mouse_id: str, device: torch.device = "cpu"
):
    results = {"predictions": [], "targets": [], "pupil_dilations": []}
    model.to(device)
    model.train(False)
    for data in tqdm(ds, desc=f"Mouse {mouse_id}"):
        predictions, _, _ = model(
            inputs=data["image"].to(device),
            mouse_id=mouse_id,
            behaviors=data["behavior"].to(device),
            pupil_centers=data["pupil_center"].to(device),
        )
        results["predictions"].append(predictions.cpu().numpy())
        results["targets"].append(data["response"].numpy())
        results["pupil_dilations"].append(data["behavior"][:, 0].numpy())
    results = {k: np.vstack(v) for k, v in results.items()}
    results["pupil_dilations"] = np.squeeze(results["pupil_dilations"], axis=-1)
    return results


def correlation_by_dilation(results: t.Dict[str, np.ndarray]):
    # sort responses according to pupil dilation
    dilation_sort = np.argsort(results["pupil_dilations"])
    predictions = results["predictions"][dilation_sort]
    targets = results["targets"][dilation_sort]
    # compute the correlation of top-third and bottom-third responses
    third = len(dilation_sort) // 3
    small = correlation(y1=predictions[:third], y2=targets[:third], dim=0)
    large = correlation(y1=predictions[-third:], y2=targets[-third:], dim=0)
    overall = correlation(y1=predictions, y2=targets, dim=0)
    print(
        f"Overall {overall.mean():.04f}, "
        f"large: {large.mean():.04f}, "
        f"small: {small.mean():.04f}"
    )
    return {"large": large, "small": small}


def plot_correlations(
    results: t.Dict[str, t.Dict[str, np.ndarray]], filename: str = None
):
    df = pd.DataFrame(
        data=[
            [i, results[mouse_id][size][i], mouse_id, size]
            for size in ["large", "small"]
            for mouse_id in results.keys()
            for i in range(len(results[mouse_id][size]))
        ],
        columns=["neuron", "Correlation", "Mouse", "Pupil size"],
    )
    tick_fontsize, label_fontsize = 8, 10
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5), dpi=240)
    sns.violinplot(
        data=df,
        x="Mouse",
        y="Correlation",
        hue="Pupil size",
        inner="quartile",
        split=True,
        palette="Set2",
        ax=ax,
    )

    sns.despine(ax=ax, offset={"left": 15, "bottom": 5}, trim=True)
    ax.set_yticklabels(ax.get_yticks().round(1), fontsize=tick_fontsize)
    ax.set_xticklabels(
        [data.convert_id(mouse_id) for mouse_id in results.keys()],
        fontsize=tick_fontsize,
    )
    ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
    ax.set_xlim(-0.45, 4.45)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=[plt.plot([], marker="", ls="")[0]] + handles,
        labels=["pupil size"] + labels,
        loc="lower center",
        bbox_to_anchor=[0.5, -0.04],
        frameon=False,
        ncols=3,
        prop={"size": tick_fontsize},
        handletextpad=0.4,
        columnspacing=1.0,
    )

    max_value = 1.06
    for i, mouse_id in enumerate(results.keys()):
        small = np.mean(results[mouse_id]["small"])
        large = np.mean(results[mouse_id]["large"])
        gain = 100 * (large - small) / small
        ax.text(
            x=i,
            y=max_value,
            s=f"{'+' if gain > 0 else '-'}{gain:.01f}%",
            ha="center",
            va="top",
            fontsize=tick_fontsize,
        )
    # ax.set_title("Prediction performance w.r.t pupil dilation", fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()
    if filename is not None:
        tensorboard.save_figure(figure=figure, filename=filename, dpi=240)
        print(f"Figure saved at {filename}.")


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

    # results = {}
    # for mouse_id, mouse_ds in test_ds.items():
    #     if mouse_id == "1":
    #         continue
    #     mouse_result = inference(
    #         model=model, ds=mouse_ds, mouse_id=mouse_id, device=args.device
    #     )
    #     correlations = correlation_by_dilation(mouse_result)
    #     results[mouse_id] = correlations

    import pickle

    with open(os.path.join(args.output_dir, "pupil_dilation.pkl"), "rb") as file:
        results = pickle.load(file)

    plot_correlations(
        results,
        filename=os.path.join(
            args.output_dir, "plots", "pupil_dilation_correlation.svg"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    main(parser.parse_args())
