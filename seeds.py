import os
import torch
import pickle
import argparse
from copy import deepcopy
from shutil import rmtree
from os.path import abspath
import numpy as np
from sensorium.utils import yaml

import train as trainer


class Args:
    def __init__(self, trial_id: int, output_dir: str, seed: int, config: dict):
        self.output_dir = os.path.join(output_dir, f"{trial_id:03d}_s{seed}")
        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def train_function(args, trial_id: int, seed: int, config: dict):
    params = Args(
        trial_id=trial_id,
        output_dir=args.output_dir,
        seed=seed,
        config=config,
    )
    print(f"Trial {trial_id}: {params.output_dir}")
    return trainer.main(params)


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    config = {
        "dataset": abspath(args.dataset),
        "output_dir": abspath(os.path.join(args.output_dir, "output_dir")),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "core": "stacked2d",
        "readout": "gaussian2d",
        "mouse_ids": None,
        "num_workers": args.num_workers,
        "plus": False,
        "depth_scale": 1,
        "device": args.device,
        "pretrain_core": "",
        "save_plots": False,
        "dpi": 120,
        "format": "svg",
        "clear_output_dir": False,
        "verbose": 0,
        "num_layers": 4,
        "dropout": 0.0,
        "disable_grid_predictor": False,
        "grid_predictor_dim": 2,
        "bias_mode": 0,
        "criterion": "poisson",
        "lr": 1e-3,
        "core_reg_scale": 0,
        "readout_reg_scale": 0.0076,
        "ds_scale": True,
        "crop_mode": 1,
        "core_lr_scale": 1,
        "use_shifter": False,
    }

    results = []
    for seed in range(args.num_samples):
        config["seed"] = seed
        result = train_function(args, trial_id=seed, seed=seed, config=deepcopy(config))
        results.append(result["single_trial_correlation"])

    best_seed = np.argmax(results)
    best_corr = results[best_seed]
    yaml.save(
        filename=os.path.join(args.output_dir, "result.yaml"),
        data={"results": results, "best_seed": best_seed, "best_corr": best_corr},
    )
    print(f"\n\nBest seed {best_seed} with single trial correlation {best_corr}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="data",
        help="path to directory where the compressed dataset is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_workers", default=2, type=int, help="number of works for DataLoader."
    )

    # model settings
    parser.add_argument("--batch_size", type=int, default=0)

    # training settings
    parser.add_argument(
        "--epochs", default=200, type=int, help="maximum epochs to train the model."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="",
        help="Device to use for computation. "
        "Use the best available device if --device is not specified.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_samples", type=int, default=100)

    # misc
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])

    main(parser.parse_args())
