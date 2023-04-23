import os
import torch
import pickle
import argparse
import warnings
import numpy as np
import typing as t
from tqdm import tqdm
import scipy.optimize as opt
from einops import rearrange, einsum

from v1t import data
from v1t.utils import utils
from v1t.models.model import Model
from v1t.utils.scheduler import Scheduler
from torch.utils.data import TensorDataset, DataLoader


warnings.simplefilter("error", opt.OptimizeWarning)
IMAGE_SIZE = (1, 36, 64)


def normalize(image: t.Union[np.array, torch.Tensor]):
    """Normalize image to [0, 1] using its min and max values."""
    return (image - image.min()) / (image.max() - image.min())


def load_model(args) -> Model:
    train_ds, _, _ = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )
    model = Model(args, ds=train_ds)
    model.to(args.device)
    scheduler = Scheduler(args, model=model, save_optimizer=False)
    _ = scheduler.restore(force=True)
    return model


def generate_ds(args, num_samples: int):
    """Generate num_samples of white noise images from uniform distribution
    Return:
        ds: DataLoader, the DataLoader object with the white noise images
        noise: np.ndarray, the array with the raw white noise images
    """
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
def inference(args, model: Model, ds: DataLoader) -> torch.Tensor:
    responses = []
    device, mouse_id = args.device, "A"
    for batch in tqdm(ds, desc=f"Mouse {mouse_id}"):
        images = batch[0].to(device)
        batch_size = images.size(0)
        # create dummy behaviors to match input arguments
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
        responses.append(outputs.cpu())
    responses = torch.concat(responses, dim=0)
    return responses


def estimate_RFs(activations: torch.Tensor, noise: torch.Tensor) -> np.ndarray:
    """
    Compute sum of the white noise images weighted their corresponding
    response value to estimate the artificial RFs
    """
    aRFs = einsum(activations, noise, "b n, b c h w -> n c h w")
    return aRFs.numpy()


def Gaussian2d(
    xy: np.ndarray,
    amplitude: float,
    xo: float,
    yo: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
):

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


def fit_gaussian(args, aRFs: np.ndarray) -> np.ndarray:
    """Fit 2D Gaussian to each aRFs using SciPy curve_fit

    Gaussian fit reference: https://stackoverflow.com/a/21566831

    Returns:
        popts: np.ndarray, a (num. units, 7) array with fitted parameters in
            [amplitude, center x, center y, sigma x, sigma y, theta, offset]
    """
    num_units = aRFs.shape[0]

    # standardize RFs and take absolute values to remove background noise
    mean = np.mean(aRFs, axis=(1, 2, 3))
    std = np.std(aRFs, axis=(1, 2, 3))
    broadcast = lambda a: rearrange(a, "n -> n 1 1 1")
    aRFs = (aRFs - broadcast(mean)) / broadcast(std)
    aRFs = np.abs(aRFs)

    height, width = aRFs.shape[2:]
    x, y = np.linspace(0, width, width), np.linspace(0, height, height)
    x, y = np.meshgrid(x, y)

    # numpy array of optimal parameters where rows are unit index
    popts = np.full(shape=(num_units, 7), fill_value=np.inf, dtype=np.float32)
    for i, unit in enumerate(tqdm(range(num_units), desc="Fit 2D Gaussian")):
        data = aRFs[unit][0]
        data = data.ravel()
        data_noisy = data + 0.2 * np.random.normal(size=data.shape)
        try:
            popt, pcov = opt.curve_fit(
                f=Gaussian2d,
                xdata=(x, y),
                ydata=data_noisy,
                p0=(3, width // 2, height // 2, 10, 10, 0, 10),
            )
            popts[unit] = popt
        except (RuntimeError, opt.OptimizeWarning):
            pass

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

    return popts


def main(args):
    utils.get_device(args)
    utils.set_random_seed(args.seed)

    filename = os.path.join(args.output_dir, "aRFs.pkl")

    results = {}
    if os.path.exists(filename):
        print(f"Load aRFs from {filename}.")
        with open(filename, "rb") as file:
            results = pickle.load(file)

    if "aRFs" not in results or args.overwrite:
        utils.load_args(args)
        model = load_model(args)

        ds, noise = generate_ds(args, num_samples=args.sample_size)
        activations = inference(args, model=model, ds=ds)

        aRFs = estimate_RFs(activations=activations, noise=noise)

        results["aRFs"] = aRFs
    else:
        aRFs = results["aRFs"]

    results["popts"] = fit_gaussian(args, aRFs=aRFs)

    with open(filename, "wb") as file:
        pickle.dump(results, file)

    print(f"Saved aRFs and Gaussian fits to {filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument("--sample_size", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    main(parser.parse_args())
