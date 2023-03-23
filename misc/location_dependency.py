import os
import torch
import argparse
import platform
import numpy as np
from tqdm import tqdm
from PIL import ImageColor
import matplotlib
import matplotlib.pyplot as plt
import typing as t
from v1t import data
from v1t.models.model import Model
from v1t.utils import utils, tensorboard
from v1t.utils.scheduler import Scheduler
from torch.utils.data import TensorDataset, DataLoader
from v1t.models import Model
import pickle
from einops import rearrange, repeat, einsum
from torch import nn
from torch.nn import functional as F

from v1t.fitgabor import GaborGenerator, trainer_fn

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
    device, mouse_id = args.device, args.mouse_ids[0]
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
    results = torch.concat(results, dim=0)
    # reshape activations from (N, C, H, W) to (N, H*W, C)
    # results = rearrange(results, "b c h w -> b (h w) c")
    # compute activations in the hidden dimension by average
    # activations = torch.mean(results, dim=-1)
    activations = results
    return activations


def compute_weighted_RFs(args, activations: torch.tensor, noise: torch.tensor):
    # num_units = activations.size(1)
    # weighted_activations = torch.zeros(
    #     (num_units,) + noise.shape[1:], device=args.device
    # )
    #
    # for unit in tqdm(range(num_units), desc="Compute activations"):
    #     activation = activations[:, unit]
    #     weighted_activation = noise * repeat(activation, "b -> b 1 1 1")
    #     weighted_activations[unit] = torch.sum(weighted_activation, dim=0)
    RFs = einsum(activations, noise, "b n, b c h w -> n c h w")
    return RFs


def normalize(image: t.Union[np.array, torch.tensor]):
    if torch.is_tensor(image):
        i_min, i_max = torch.min(image), torch.max(image)
    else:
        i_min, i_max = np.min(image), np.max(image)
    return (image - i_min) / (i_max - i_min)


def plot_grid(args, weighted_RFs: t.Union[torch.tensor, np.array], title: str = None):
    images = weighted_RFs
    if torch.is_tensor(images):
        images = weighted_RFs.numpy()

    nrows, ncols = 5, 3
    max_images = nrows * ncols

    num_units = images.shape[0]

    # randomly select max_images to plot
    units = sorted(np.random.choice(num_units, size=max_images, replace=False))

    tick_fontsize, label_fontsize, title_fontsize = 8, 9, 10
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        gridspec_kw={"wspace": 0.02, "hspace": 0.2},
        figsize=(6, 7),
        dpi=120,
    )

    for i, ax in enumerate(axes.flat):
        unit = units[i]
        ax.imshow(normalize(images[unit][0]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Unit #{unit}", pad=0, fontsize=label_fontsize)
        ax.axis("off")

    if title:
        pos = axes[0, 1].get_position()
        figure.suptitle(title, fontsize=title_fontsize, y=pos.y1 * 1.05)

    plt.show()

    if title:
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


import scipy.optimize as opt


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
    # plot_grid(args, weighted_RFs=weighted_RFs)
    height, width = weighted_RFs.shape[2:]
    x, y = np.linspace(0, width, width), np.linspace(0, height, height)
    x, y = np.meshgrid(x, y)

    nrows, ncols = 5, 3
    max_images = nrows * ncols
    units = sorted(np.random.choice(num_units, size=max_images, replace=False))
    tick_fontsize, label_fontsize, title_fontsize = 8, 9, 10
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        gridspec_kw={"wspace": 0.02, "hspace": 0.2},
        figsize=(6, 7),
        dpi=120,
    )
    axes = axes.flatten()

    sigma_x = []
    sigma_y = []
    standard_deviations = []
    plot_idx = 0
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
            sigma_x.append(popt[3])
            sigma_y.append(popt[4])
            if unit in units:
                fitted = Gaussian2d((x, y), *popt)
                axes[plot_idx].imshow(
                    normalize(data.reshape(height, width)), cmap="gray", vmin=0, vmax=1
                )
                axes[plot_idx].contour(
                    x, y, fitted.reshape(height, width), z=8, alpha=0.4, colors="red"
                )
                axes[plot_idx].set_title(
                    f"Unit #{unit}", pad=0, fontsize=label_fontsize
                )
                plot_idx += 1
        except RuntimeError:
            print(f"Cannot find optimal parameters for unit {unit}.")

    for ax in axes:
        ax.axis("off")

    sigma_x, sigma_y = np.array(sigma_x), np.array(sigma_y)
    print(
        f"sigma X: {np.mean(sigma_x):.04f} \pm {np.std(sigma_x):.04f}\n"
        f"sigma Y: {np.mean(sigma_y):.04f} \pm {np.std(sigma_y):.04f}"
    )

    plt.show()
    filename = os.path.join(args.output_dir, "plots", "fit_gaussian.jpg")
    tensorboard.save_figure(figure, filename=filename, dpi=240, close=True)
    print(f"Saved plot to {filename}.")


def main(args):
    utils.set_random_seed(1234)
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
    plot_grid(
        args,
        weighted_RFs=weighted_RFs,
        title="ViT RFs" if "vit" in args.output_dir else "CNN RFs",
    )
    # fit_gabor(args, weighted_RFs=weighted_RFs)
    fit_gaussian(args, weighted_RFs=weighted_RFs)

    print(f"Results saved to {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
