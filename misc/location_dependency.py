import os
import torch
import argparse
import platform
import numpy as np
from tqdm import tqdm
from PIL import ImageColor
import matplotlib
import matplotlib.pyplot as plt

from v1t import data
from v1t.models.model import Model
from v1t.utils import utils, tensorboard
from v1t.utils.scheduler import Scheduler
from torch.utils.data import TensorDataset, DataLoader
from v1t.models import Model
import pickle
from einops import rearrange, repeat, einsum


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
    noise = torch.rand((num_samples, 1, 36, 64), device=args.device)
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

        results.append(outputs)
    results = torch.concat(results, dim=0)
    # reshape activations from (N, C, H, W) to (N, H*W, C)
    # results = rearrange(results, "b c h w -> b (h w) c")
    # compute activations in the hidden dimension by average
    # activations = torch.mean(results, dim=-1)
    activations = results
    return activations


def compute_weighted_activations(args, activations: torch.tensor, noise: torch.tensor):
    # num_units = activations.size(1)
    # weighted_activations = torch.zeros(
    #     (num_units,) + noise.shape[1:], device=args.device
    # )
    #
    # for unit in tqdm(range(num_units), desc="Compute activations"):
    #     activation = activations[:, unit]
    #     weighted_activation = noise * repeat(activation, "b -> b 1 1 1")
    #     weighted_activations[unit] = torch.sum(weighted_activation, dim=0)
    weighted_activations = einsum(activations, noise, "b n, b c h w -> n c h w")
    return weighted_activations.numpy()


def normalize(image: np.ndarray):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def plot_grid(args, weighted_activations: np.ndarray):
    max_images = 9

    # randomly select 9 units
    num_units = weighted_activations.shape[0]
    unit_idx = np.random.choice(num_units, size=max_images, replace=False)

    tick_fontsize, label_fontsize = 8, 9
    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        gridspec_kw={"wspace": 0.02, "hspace": 0.2},
        figsize=(8, 4),
        dpi=120,
    )

    for i, ax in enumerate(axes.flat):
        unit = unit_idx[i]
        image = weighted_activations[unit][0]
        ax.imshow(normalize(image), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Unit #{unit}", pad=0, fontsize=label_fontsize)
        ax.axis("off")

    plt.show()
    # plt.close(figure)


def main(args):
    utils.set_random_seed(1234)
    args.device = torch.device(args.device)

    utils.load_args(args)
    model = load_model(args)

    ds, noise = generate_ds(args, num_samples=5000)
    activations = inference(args, model=model, ds=ds)

    weighted_activations = compute_weighted_activations(
        args, activations=activations, noise=noise
    )

    filename = os.path.join(args.output_dir, "weighted_activations.pkl")
    with open(filename, "w") as file:
        pickle.dump(weighted_activations, file)

    plot_grid(args, weighted_activations)

    print(f"results saved to {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    main(parser.parse_args())
