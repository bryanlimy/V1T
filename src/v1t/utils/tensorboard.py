import os
import io
import torch
import platform
import matplotlib
import numpy as np
import typing as t
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.font_manager as font_manager
from torch.utils.tensorboard import SummaryWriter


sns.set_style("ticks")
plt.style.use("seaborn-deep")

PARAMS_PAD = 2
PARAMS_LENGTH = 3

plt.rcParams.update(
    {
        "mathtext.default": "regular",
        "xtick.major.pad": PARAMS_PAD,
        "ytick.major.pad": PARAMS_PAD,
        "xtick.major.size": PARAMS_LENGTH,
        "ytick.major.size": PARAMS_LENGTH,
    }
)

TICKER_FORMAT = matplotlib.ticker.FormatStrFormatter("%.2f")

JET = cm.get_cmap("jet")
GRAY = cm.get_cmap("gray")
TURBO = cm.get_cmap("turbo")
COLORMAP = TURBO
GRAY2RGB = COLORMAP(np.arange(256))[:, :3]


def set_font():
    font_path = os.getenv("MATPLOTLIB_FONT")
    if font_path is not None and os.path.exists(font_path):
        font_manager.fontManager.addfont(path=font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams.update(
            {"font.family": "sans-serif", "font.sans-serif": prop.get_name()}
        )


def remove_spines(axis: matplotlib.axes.Axes):
    """remove all spines"""
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)


def remove_top_right_spines(axis: matplotlib.axes.Axes):
    """Remove the ticks and spines of the top and right axis"""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def set_right_label(axis: matplotlib.axes.Axes, label: str, fontsize: int = None):
    """Set y-axis label on the right-hand side"""
    right_axis = axis.twinx()
    kwargs = {"rotation": 270, "va": "center", "labelpad": 3}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    right_axis.set_ylabel(label, **kwargs)
    right_axis.set_yticks([])
    remove_top_right_spines(right_axis)


def set_xticks(
    axis: matplotlib.axes.Axes,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list] = None,
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_xticks(ticks_loc)
    if ticks is None:
        ticks = ticks_loc
    axis.set_xticklabels(ticks, fontsize=tick_fontsize)
    if label:
        axis.set_xlabel(label, fontsize=label_fontsize)


def set_yticks(
    axis: matplotlib.axes.Axes,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list] = None,
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_yticks(ticks_loc)
    if ticks is None:
        ticks = ticks_loc
    axis.set_yticklabels(ticks, fontsize=tick_fontsize)
    if label:
        axis.set_ylabel(label, fontsize=label_fontsize)


def set_ticks_params(
    axis: matplotlib.axes.Axes, length: int = PARAMS_LENGTH, pad: int = PARAMS_PAD
):
    axis.tick_params(axis="both", which="both", length=length, pad=pad, colors="black")


def save_figure(figure: plt.Figure, filename: str, dpi: int = 120, close: bool = True):
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    figure.savefig(
        filename, dpi=dpi, bbox_inches="tight", pad_inches=0.01, transparent=True
    )
    if close:
        plt.close(figure)


class Summary(object):
    """Helper class to write TensorBoard summaries"""

    def __init__(self, args):
        self.dpi = args.dpi
        self.format = args.format
        self.save_plots = args.save_plots

        # create SummaryWriter for train, validation and test set
        self.writers = [
            SummaryWriter(args.output_dir),
            SummaryWriter(os.path.join(args.output_dir, "val")),
            SummaryWriter(os.path.join(args.output_dir, "test")),
        ]

        self.plots_dir = os.path.join(args.output_dir, "plots")
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)

        if platform.system() == "Darwin" and args.verbose > 2:
            matplotlib.use("TkAgg")

    def get_writer(self, mode: int = 0):
        """Get SummaryWriter
        Args:
            mode: int, the SummaryWriter to get
                0 - train set
                1 - validation set
                2 - test set
        """
        return self.writers[mode]

    def close(self):
        for writer in self.writers:
            writer.close()

    def scalar(self, tag, value, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        writer.add_scalar(tag, scalar_value=value, global_step=step)

    def histogram(self, tag, values, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        writer.add_histogram(tag, values=values, global_step=step)

    def image(self, tag, values, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        writer.add_image(tag, img_tensor=values, global_step=step, dataformats="CHW")

    def figure(
        self,
        tag: str,
        figure: plt.Figure,
        step: int = 0,
        close: bool = True,
        mode: int = 0,
    ):
        """Write matplotlib figure to summary
        Args:
          tag: str, data identifier
          figure: plt.Figure, matplotlib figure or a list of figures
          step: int, global step value to record
          close: bool, close figure if True
          mode: int, indicate which summary writers to use
        """
        if self.save_plots:
            save_figure(
                figure,
                filename=os.path.join(
                    self.plots_dir, f"epoch_{step:03d}", f"{tag}.{self.format}"
                ),
                dpi=self.dpi,
                close=False,
            )
        buffer = io.BytesIO()
        figure.savefig(
            buffer, dpi=self.dpi, format="png", bbox_inches="tight", pad_inches=0.02
        )
        buffer.seek(0)
        image = Image.open(buffer)
        image = transforms.ToTensor()(image)
        self.image(tag, image, step=step, mode=mode)
        if close:
            plt.close(figure)

    def box_plot(
        self,
        tag: str,
        data: pd.DataFrame,
        xlabel: str = "Mouse",
        ylabel: str = "Correlation",
        step: int = 0,
        mode: int = 0,
    ):
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=self.dpi)
        sns.boxenplot(x="mouse", y="results", data=data, ax=ax)
        sns.despine(trim=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.figure(tag, figure=figure, step=step, close=True, mode=mode)

    def plot_image_response(
        self,
        tag: str,
        results: t.Dict[str, np.ndarray],
        step: int = 0,
        mode: int = 1,
    ):
        """Plot image-prediction-response for each mouse"""
        num_samples = len(results["images"])
        wspace, hspace = 0.1, 1.0
        label_fontsize, tick_fontsize = 10, 9
        figure = plt.figure(figsize=(10, 1.8 * num_samples), dpi=self.dpi)
        sub_figures = figure.subfigures(nrows=num_samples, ncols=1, hspace=hspace)
        num_neurons = results["predictions"].shape[1]
        x_axis = np.arange(num_neurons)

        # the (x, y) coordinates in crop_grids are in range [-1, 1]
        # need to convert to [0, 144] and [0, 256] in height and width
        # matplotlib assume top left corner of the image is (0, 0)
        _, _, h, w = results["images"].shape
        image_grids = [w, h] * (results["image_grids"] + 1) / 2
        image_grids = np.round(image_grids, 0)

        for i in range(num_samples):
            axes = sub_figures[i].subplots(
                nrows=1,
                ncols=4,
                gridspec_kw={
                    "width_ratios": [0.5, 0.5, 1, 1],
                    "wspace": wspace,
                    "hspace": hspace,
                },
            )
            image = results["images"][i]
            crop_image = results["crop_images"][i]
            image_grid = image_grids[i]
            target = results["targets"][i]
            prediction = results["predictions"][i]
            pupil_center = results["pupil_center"][i]
            behavior = results["behaviors"][i]
            axes[0].scatter(
                x=x_axis,
                y=target,
                s=2,
                alpha=0.8,
                color="orangered",
                label="target",
            )
            axes[1].scatter(
                x=x_axis,
                y=prediction,
                s=2,
                alpha=0.8,
                color="dodgerblue",
                label="prediction",
            )
            y_max = np.ceil(max(np.max(target), np.max(prediction)))
            set_yticks(
                axes[0],
                ticks_loc=np.linspace(0, y_max, 3, dtype=int),
                tick_fontsize=tick_fontsize,
            )
            axes[0].set_ylim(0, y_max)
            axes[1].set_ylim(0, y_max)
            axes[1].set_yticks([])
            remove_top_right_spines(axis=axes[0])
            remove_top_right_spines(axis=axes[1])

            # plot image
            axes[2].imshow(image[0], cmap=GRAY, vmin=0, vmax=255)
            # overlay the cropping grid as a red rectangle box
            axes[2].add_patch(
                matplotlib.patches.Rectangle(
                    image_grid[0, 0],
                    width=image_grid.shape[1],
                    height=image_grid.shape[0],
                    alpha=1,
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2,
                )
            )
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            remove_spines(axis=axes[2])
            axes[3].imshow(crop_image[0], cmap=GRAY, vmin=0, vmax=255)
            axes[3].set_xticks([])
            axes[3].set_yticks([])
            remove_spines(axis=axes[3])

            sub_figures[i].suptitle(
                f"Image ID: {results['image_ids'][i]}\n"
                f"pupil dilation: {behavior[0]:.02f}, "
                f"derivative: {behavior[1]:.02f}, "
                f"speed: {behavior[2]:.02f}, "
                f"pupil center: ({pupil_center[0]:.02f}, {pupil_center[1]:.02f})",
                y=1.05,
                fontsize=label_fontsize,
            )
            if i == 0:
                axes[0].legend(
                    frameon=False, handletextpad=0.3, handlelength=0.6, markerscale=2
                )
                axes[1].legend(
                    frameon=False, handletextpad=0.3, handlelength=0.6, markerscale=2
                )
            if i == num_samples // 2:
                axes[0].set_ylabel("Standardized responses", fontsize=label_fontsize)
            if i == num_samples - 1:
                x_ticks = np.linspace(0, num_neurons, num=3, dtype=int)
                set_xticks(
                    axis=axes[0],
                    ticks_loc=x_ticks,
                    tick_fontsize=tick_fontsize,
                )
                set_xticks(
                    axis=axes[1],
                    ticks_loc=x_ticks,
                    label="Neurons",
                    tick_fontsize=tick_fontsize,
                    label_fontsize=label_fontsize,
                )
                axes[2].set_xlabel("Model input", fontsize=label_fontsize)
                axes[3].set_xlabel("Core input", fontsize=label_fontsize)
                sub_figures[i].align_ylabels(axes)
            else:
                axes[0].set_xticks([])
                axes[1].set_xticks([])
            for ax in axes:
                set_ticks_params(axis=ax, pad=1, length=2)

        self.figure(
            tag=tag,
            figure=figure,
            step=step,
            close=True,
            mode=mode,
        )
