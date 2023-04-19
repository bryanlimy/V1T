import os
import argparse
import matplotlib
import numpy as np
import typing as t
import matplotlib.cm as cm
import matplotlib.pyplot as plt


from v1t import data
from v1t.models.model import Model
from v1t.utils import utils, tensorboard
from v1t.utils.scheduler import Scheduler
from v1t.utils.attention_rollout import extract_attention_maps


def to_rgb(image: np.ndarray):
    assert image.shape[0] == 2
    blue, green = image[0][..., None], image[1][..., None]
    red = np.zeros_like(blue)
    return np.concatenate((red, blue, green), axis=-1)


def plot_attention_map(
    results: t.Dict[str, np.ndarray],
    filename: str = None,
    colormap: str = "turbo",
    alpha: float = 0.5,
):
    cmap = matplotlib.colormaps.get_cmap(colormap)
    colors = cmap(np.arange(256))[:, :3]
    label_fontsize, tick_fontsize = 10, 8
    figure, axes = plt.subplots(
        nrows=len(results["images"]),
        ncols=2,
        figsize=(7, 2.2 * len(results["images"])),
        gridspec_kw={"wspace": 0.05, "hspace": 0.2},
        dpi=120,
    )
    for i in range(len(results["images"])):
        image = results["images"][i]
        heatmap = results["heatmaps"][i]
        behavior = results["behaviors"][i]
        pupil_center = results["pupil_centers"][i]
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
        tensorboard.save_figure(figure, filename=filename, dpi=120)
        print(f"plot saved to {filename}.")


def plot_attention_map_2(
    val_results: t.Dict[str, np.ndarray],
    test_results: t.Dict[str, np.ndarray],
    filename: str = None,
    colormap: str = "turbo",
    alpha: float = 0.5,
):
    cmap = matplotlib.colormaps.get_cmap(colormap)
    colors = cmap(np.arange(256))[:, :3]
    label_fontsize, tick_fontsize = 10, 8
    x_labelpad, y_labelpad = 2, 1
    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(8, 4),
        gridspec_kw={"wspace": 0.05, "hspace": -0.25},
        dpi=240,
    )
    for i in range(len(val_results["images"])):
        image = val_results["images"][i]
        heatmap = val_results["heatmaps"][i]
        behavior = val_results["behaviors"][i]
        pupil_center = val_results["pupil_centers"][i]
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
        axes[0, i].set_xlabel(description, labelpad=x_labelpad, fontsize=tick_fontsize)
    axes[0, 0].set_ylabel(
        "Validation samples", labelpad=y_labelpad, fontsize=tick_fontsize
    )
    for i in range(len(test_results["images"])):
        image = test_results["images"][i]
        heatmap = test_results["heatmaps"][i]
        behavior = test_results["behaviors"][i]
        pupil_center = test_results["pupil_centers"][i]
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
        axes[1, i].set_xlabel(description, labelpad=x_labelpad, fontsize=tick_fontsize)
    axes[-1, 0].set_ylabel("Test samples", labelpad=y_labelpad, fontsize=tick_fontsize)

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


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")
    tensorboard.set_font()

    utils.get_device(args)
    utils.set_random_seed(1234)

    utils.load_args(args)

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

    mouse_id = "A"

    val_results = extract_attention_maps(
        val_ds[mouse_id], model=model, num_samples=3, device=args.device
    )
    test_results = extract_attention_maps(
        test_ds[mouse_id], model=model, num_samples=3, device=args.device
    )

    plot_dir = os.path.join(args.output_dir, "plots")
    plot_attention_map(
        results=val_results,
        filename=os.path.join(plot_dir, f"attention_rollout_mouse{mouse_id}.jpg"),
    )
    plot_attention_map_2(
        val_results=val_results,
        test_results=test_results,
        filename=os.path.join(plot_dir, f"attention_rollout_mouse{mouse_id}.jpg"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)

    main(parser.parse_args())
