import os
import io
import platform
import matplotlib
import numpy as np
import typing as t
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
    ticks: t.Union[np.ndarray, list],
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_xticks(ticks_loc)
    axis.set_xticklabels(ticks, fontsize=tick_fontsize)
    if label:
        axis.set_xlabel(label, fontsize=label_fontsize)


def set_yticks(
    axis: matplotlib.axes.Axes,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list],
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_yticks(ticks_loc)
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

    def __init__(self, args, output_dir: str = ""):
        self.dpi = args.dpi
        self.format = args.format
        self.dataset = args.dataset
        self.save_plots = args.save_plots

        # write TensorBoard summary to specified output_dir or args.output_dir
        if output_dir:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.writers = [SummaryWriter(output_dir)]
        else:
            output_dir = args.output_dir
            self.writers = [
                SummaryWriter(output_dir),
                SummaryWriter(os.path.join(output_dir, "val")),
                SummaryWriter(os.path.join(output_dir, "test")),
            ]

        self.plots_dir = os.path.join(output_dir, "plots")
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)

        if platform.system() == "Darwin" and args.verbose == 2:
            matplotlib.use("TkAgg")

    def get_writer(self, mode: int = 0):
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
