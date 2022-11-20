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
        label_fontsize, tick_fontsize = 12, 10
        figure, axes = plt.subplots(
            nrows=num_samples,
            ncols=3,
            gridspec_kw={"wspace": 0.05, "hspace": 0.4},
            figsize=(10, 2 * num_samples),
            dpi=self.dpi,
        )
        num_neurons = results["predictions"].shape[1]
        x_axis = np.arange(num_neurons)

        for i in range(num_samples):
            image = results["images"][i]
            target = results["targets"][i]
            prediction = results["predictions"][i]
            pupil_center = results["pupil_center"][i]
            axes[i, 0].scatter(x=x_axis, y=target, s=2, alpha=0.8, color="orangered")
            axes[i, 1].scatter(
                x=x_axis, y=prediction, s=2, alpha=0.8, color="dodgerblue"
            )
            y_max = np.ceil(max(np.max(target), np.max(prediction)))
            set_yticks(
                axes[i, 0],
                ticks_loc=[0, y_max],
                tick_fontsize=tick_fontsize,
            )
            axes[i, 0].set_ylim(0, y_max)
            axes[i, 1].set_ylim(0, y_max)
            axes[i, 1].set_yticks([])
            if i == num_samples - 1:
                x_ticks = np.linspace(0, num_neurons, num=3, dtype=int)
                set_xticks(
                    axis=axes[i, 0],
                    ticks_loc=x_ticks,
                    tick_fontsize=tick_fontsize,
                )
                set_xticks(
                    axis=axes[i, 1],
                    ticks_loc=x_ticks,
                    tick_fontsize=tick_fontsize,
                )
            else:
                axes[i, 0].set_xticks([])
                axes[i, 1].set_xticks([])
            axes[i, 2].imshow(image[0], cmap=GRAY, vmin=0, vmax=255)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            remove_top_right_spines(axis=axes[i, 0])
            remove_top_right_spines(axis=axes[i, 1])
            remove_spines(axis=axes[i, 2])
            axes[i, 2].set_xlabel(
                f"Image ID: {results['image_ids'][i]}\n"
                f"Pupil Center: [{pupil_center[0]:.02f}, {pupil_center[1]:.02f}]",
                labelpad=0,
                fontsize=tick_fontsize,
            )

        axes[0, 0].set_title("Targets", fontsize=label_fontsize)
        axes[0, 1].set_title("Predictions", fontsize=label_fontsize)
        axes[num_samples // 2, 0].set_ylabel("Response", fontsize=label_fontsize)
        axes[num_samples - 1, 1].set_xlabel("Neurons", fontsize=label_fontsize)

        self.figure(
            tag=tag,
            figure=figure,
            step=step,
            close=True,
            mode=mode,
        )
