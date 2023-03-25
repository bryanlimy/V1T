import os
import torch
import pickle
import argparse
import warnings
import numpy as np
import typing as t
from torch import nn
from tqdm import tqdm
import scipy.optimize as opt
import matplotlib.pyplot as plt
from torch.nn import functional as F
from einops import rearrange, einsum

from v1t import data
from v1t.models.model import Model
from v1t.utils import utils, tensorboard
from v1t.utils.scheduler import Scheduler
from v1t.fitgabor import GaborGenerator, trainer_fn
from torch.utils.data import TensorDataset, DataLoader


warnings.simplefilter("error", opt.OptimizeWarning)
IMAGE_SIZE = (1, 36, 64)


def load_model(args):
    _, val_ds, _ = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )
    model = Model(args, ds=val_ds)
    model.to(args.device)
    scheduler = Scheduler(args, model=model, save_optimizer=False)
    _ = scheduler.restore(force=True)
    return model


def generate_ds(args, num_samples: int = 5000):
    noise = torch.rand((num_samples, *IMAGE_SIZE))
    # standardize images
    mean, std = torch.mean(noise), torch.std(noise)
    images = (noise - mean) / std
    # create DataLoader
    dataloader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    if args.device.type in ("cuda", "mps"):
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)
    ds = DataLoader(TensorDataset(images), shuffle=False, **dataloader_kwargs)
    return ds, noise


@torch.no_grad()
def inference(args, model: Model, ds: DataLoader):
    results = []
    device, mouse_id = args.device, "A"
    for batch in tqdm(ds, desc="Inference"):
        images = batch[0].to(device)
        batch_size = images.size(0)
        behaviors = torch.zeros((batch_size, 3), device=device)
        pupil_centers = torch.zeros((batch_size, 2), device=device)
        # run model without image cropper
        outputs = model.core(
            images,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        outputs = model.readouts(outputs, mouse_id=mouse_id, shifts=None)
        outputs = model.elu1(outputs)

        results.append(outputs.cpu())
    activations = torch.concat(results, dim=0)
    return activations


def compute_weighted_RFs(args, activations: torch.tensor, noise: torch.tensor):
    # num_units = activations.size(1)
    # RFs = torch.zeros(
    #     (num_units,) + noise.shape[1:], device=args.device
    # )
    #
    # for unit in tqdm(range(num_units), desc="Compute activations"):
    #     activation = activations[:, unit]
    #     RF = noise * repeat(activation, "b -> b 1 1 1")
    #     RFs[unit] = torch.sum(RF, dim=0)
    RFs = einsum(activations, noise, "b n, b c h w -> n c h w")
    return RFs


def normalize(image: t.Union[np.array, torch.tensor]):
    if torch.is_tensor(image):
        i_min, i_max = torch.min(image), torch.max(image)
    else:
        i_min, i_max = np.min(image), np.max(image)
    return (image - i_min) / (i_max - i_min)


def plot_grid(args, weighted_RFs: t.Union[torch.tensor, np.array]):
    images = weighted_RFs
    if torch.is_tensor(images):
        images = weighted_RFs.numpy()

    nrows, ncols = 6, 3

    tick_fontsize, label_fontsize, title_fontsize = 8, 9, 10
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        gridspec_kw={"wspace": 0.02, "hspace": 0.2},
        figsize=(6, 8),
        dpi=120,
    )

    for i, ax in enumerate(axes.flat):
        unit = args.random_units[i]
        ax.imshow(normalize(images[unit][0]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Unit #{unit}", pad=0, fontsize=label_fontsize)
        ax.axis("off")

    title = "Figure C: "
    title += "ViT RFs" if "vit" in args.output_dir else "CNN RFs"
    pos = axes[0, 1].get_position()
    figure.suptitle(title, fontsize=title_fontsize, y=pos.y1 * 1.05)

    # plt.show()

    filename = os.path.join(args.output_dir, "plots", "location_filters.jpg")
    tensorboard.save_figure(figure, filename=filename, dpi=240, close=True)
    print(f"Saved weighted RFs to {filename}.")


class Neuron(nn.Module):
    def __init__(self, rf: torch.tensor):
        super(Neuron, self).__init__()
        self.rf = torch.tensor(rf, dtype=torch.float32)

    def forward(self, x):
        return F.elu((x * self.rf).sum()) + 1


def fit_gabor(args, weighted_RFs: torch.tensor):
    num_units = weighted_RFs.shape[0]
    tick_fontsize, label_fontsize = 8, 9
    sigmas = []
    for i, unit in enumerate(tqdm(range(num_units), desc="Fit Gabor")):
        ground_truth = normalize(weighted_RFs[unit][0])
        neuron = Neuron(rf=ground_truth)
        gabor_gen = GaborGenerator(image_size=IMAGE_SIZE[1:])
        gabor_gen, evolved_rfs = trainer_fn(gabor_gen, neuron, epochs=500)
        learned_gabor = gabor_gen().data.numpy()[0, 0]
        sigma = gabor_gen.sigma.data.numpy()[0]
        sigmas.append(sigma)
        if i < 10:
            figure, axes = plt.subplots(
                nrows=1,
                ncols=2,
                gridspec_kw={"wspace": 0.02, "hspace": 0.2},
                figsize=(4, 2),
                dpi=120,
            )
            axes[0].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
            axes[1].imshow(normalize(learned_gabor), cmap="gray", vmin=0, vmax=1)
            axes[0].axis("off")
            axes[1].axis("off")
            axes[0].set_title(f"Unit #{unit}", fontsize=label_fontsize)
            plt.show()
    sigmas = np.array(sigmas)
    with open(os.path.join(args.output_dir, "gabor_sigmas.pkl"), "wb") as file:
        pickle.dump(sigmas, file)
    print(f"average sigma: {np.mean(sigmas):.04f} \pm {np.std(sigmas):.04f}")


def Gaussian2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def fit_gaussian(args, weighted_RFs: torch.tensor):
    weighted_RFs = weighted_RFs.numpy()
    # standardize RFs
    mean = np.mean(weighted_RFs, axis=(1, 2, 3))
    std = np.std(weighted_RFs, axis=(1, 2, 3))
    broadcast = lambda a: rearrange(a, "n -> n 1 1 1")
    weighted_RFs = (weighted_RFs - broadcast(mean)) / broadcast(std)
    # take absolute values
    weighted_RFs = np.abs(weighted_RFs)
    num_units = weighted_RFs.shape[0]

    height, width = weighted_RFs.shape[2:]
    x, y = np.linspace(0, width, width), np.linspace(0, height, height)
    x, y = np.meshgrid(x, y)

    nrows, ncols = 6, 3
    tick_fontsize, label_fontsize, title_fontsize = 8, 9, 10
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        gridspec_kw={"wspace": 0.02, "hspace": 0.2},
        figsize=(6, 8),
        dpi=120,
    )
    axes = axes.flatten()
    ax_idx = 0  # index of the flattened axes

    # numpy array of optimal parameters where rows are unit index
    popts = np.full(shape=(num_units, 7), fill_value=np.inf, dtype=np.float32)
    for i, unit in enumerate(tqdm(range(num_units), desc="Fit 2D Gaussian")):
        data = weighted_RFs[unit][0]
        data = data.ravel()
        data_noisy = data + 0.2 * np.random.normal(size=data.shape)
        try:
            popt, pcov = opt.curve_fit(
                f=Gaussian2d,
                xdata=(x, y),
                ydata=data_noisy,
                p0=(3, width // 2, height // 2, 10, 10, 0, 10),
            )
        except (RuntimeError, opt.OptimizeWarning):
            popt, pcov = None, None

        if unit in args.random_units:
            axes[ax_idx].imshow(
                normalize(data.reshape(height, width)), cmap="gray", vmin=0, vmax=1
            )
            if popt is not None:
                fitted = Gaussian2d((x, y), *popt)
                fitted = fitted.reshape(height, width)
                axes[ax_idx].contour(
                    x,
                    y,
                    fitted,
                    levels=[1],
                    alpha=0.8,
                    linewidths=2,
                    colors="orangered",
                )
            axes[ax_idx].set_title(f"Unit #{unit}", pad=0, fontsize=label_fontsize)
            ax_idx += 1
        if popt is not None:
            popts[unit] = popt

    for ax in axes:
        ax.axis("off")

    # filter out the last 5% of the results to eliminate poor fit
    num_drops = int(0.05 * len(popts))
    large_sigma_x = np.argsort(popts[:, 3])[-num_drops:]
    large_sigma_y = np.argsort(popts[:, 4])[-num_drops:]
    drop_units = np.unique(np.concatenate((large_sigma_x, large_sigma_y), axis=0))
    popts[drop_units] = np.nan

    print(
        f"sigma X: {np.nanmean(popts[:, 3]):.03f} \pm {np.nanstd(popts[:, 3]):.03f}\n"
        f"sigma Y: {np.nanmean(popts[:, 4]):.03f} \pm {np.nanstd(popts[:, 4]):.03f}"
    )

    with open(os.path.join(args.output_dir, "gaussian_fit.pkl"), "wb") as file:
        pickle.dump(popts, file)

    title = "Figure E: "
    title += "ViT RFs" if "vit" in args.output_dir else "CNN RFs"
    title += " 2D Gaussian Fit"
    pos = axes[1].get_position()
    figure.suptitle(title, fontsize=title_fontsize, y=pos.y1 + 0.04)

    # plt.show()

    filename = os.path.join(args.output_dir, "plots", "fit_gaussian.jpg")
    tensorboard.save_figure(figure, filename=filename, dpi=240, close=True)
    print(f"Saved plot to {filename}.")


def main(args):
    utils.set_random_seed(args.seed)
    args.device = torch.device(args.device)

    filename = os.path.join(args.output_dir, "weighted_activations.pkl")

    if os.path.exists(filename) and not args.overwrite:
        print(f"Load weighted activations from {filename}.")
        with open(filename, "rb") as file:
            weighted_RFs = pickle.load(file)
    else:
        utils.load_args(args)
        model = load_model(args)

        ds, noise = generate_ds(args, num_samples=args.sample_size)
        activations = inference(args, model=model, ds=ds)

        weighted_RFs = compute_weighted_RFs(args, activations=activations, noise=noise)

        with open(filename, "wb") as file:
            pickle.dump(weighted_RFs, file)

    tensorboard.set_font()

    # randomly select 18 units to plot
    args.random_units = np.sort(
        np.random.choice(weighted_RFs.shape[0], size=18, replace=False)
    )
    plot_grid(args, weighted_RFs=weighted_RFs)
    # fit_gabor(args, weighted_RFs=weighted_RFs)
    # fit_gaussian(args, weighted_RFs=weighted_RFs)

    print(f"Results saved to {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    main(parser.parse_args())
