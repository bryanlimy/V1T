import os
import ray
import torch
import pickle
import argparse
from shutil import rmtree
from ray import air, tune
from datetime import datetime
from ray.air.config import RunConfig
from ray.tune.search.hebo import HEBOSearch
from os.path import abspath

import train as trainer


class Args:
    def __init__(self, config):
        self.output_dir = os.path.join(
            config["output_dir"], f"{datetime.now():%Y-%m-%d-%Hh%Mm}"
        )
        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


from ray.tune.utils.util import wait_for_gpu


def train_function(config):
    gpu_ids = ray.get_gpu_ids()
    # if gpu_ids:
    #     gpu_id = gpu_ids[0]
    #     while not wait_for_gpu(gpu_id):
    #         gpu_id += 1
    #         gpu_id = gpu_id % len(gpu_ids)
    #     config["device"] = f"cuda:{gpu_id}"
    args = Args(config)
    results = trainer.main(args)
    print(results)
    return results


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)

    # default search space
    search_space = {
        "dataset": abspath(args.dataset),
        "output_dir": abspath(os.path.join(args.output_dir, "output_dir")),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "readout": "gaussian2d",
        "mouse_ids": [0, 2],
        "num_workers": 2,
        "depth_scale": 1,
        "device": args.device,
        "seed": 1234,
        "save_plots": False,
        "dpi": 120,
        "format": "svg",
        "clear_output_dir": False,
        "verbose": 0,
    }

    if args.core == "vit":
        search_space.update(
            {
                "core": "vit",
                "dropout": tune.uniform(0, 0.8),
                "patch_size": tune.randint(1, 10),
                "emb_dim": tune.randint(8, 256),
                "num_heads": tune.randint(1, 16),
                "mlp_dim": tune.randint(8, 256),
                "num_layers": tune.randint(1, 8),
                "dim_head": tune.randint(8, 256),
                "disable_grid_predictor": tune.choice([True, False]),
                "grid_predictor_dim": tune.choice([2, 3]),
                "bias_mode": tune.choice([0, 1, 2]),
                "criterion": tune.choice(["rmsse", "poisson", "correlation"]),
                "lr": tune.loguniform(1e-4, 1e-2),
                "core_reg_scale": tune.loguniform(1e-5, 1),
                "readout_reg_scale": tune.loguniform(1e-5, 1),
                "ds_scale": tune.choice([True, False]),
                "crop_mode": tune.choice([0, 1, 2]),
                "pretrain_core": tune.choice(["", "runs/pretrain/001_vit_0.25dropout"]),
                "core_lr_scale": tune.loguniform(1e-4, 1),
            }
        )
        points_to_evaluate = []
    elif args.core == "stacked2d":
        search_space.update(
            {
                "core": "stacked2d",
                "dropout": tune.uniform(0, 0.8),
                "disable_grid_predictor": tune.choice([True, False]),
                "grid_predictor_dim": tune.choice([2, 3]),
                "bias_mode": tune.choice([0, 1, 2]),
                "criterion": tune.choice(["rmsse", "poisson", "correlation"]),
                "lr": tune.loguniform(1e-4, 1e-2),
                "core_reg_scale": tune.loguniform(1e-5, 1),
                "readout_reg_scale": tune.loguniform(1e-5, 1),
                "ds_scale": tune.choice([True, False]),
                "crop_mode": tune.choice([0, 1, 2]),
                "pretrain_core": tune.choice(
                    ["", "runs/pretrain/001_stacked2d_0.25dropout"]
                ),
                "core_lr_scale": tune.loguniform(1e-4, 1),
            }
        )
        points_to_evaluate = []
    elif args.core == "stn":
        search_space.update(
            {
                "core": tune.choice(["stn"]),
                "readout": tune.choice(["gaussian2d"]),
            }
        )
        points_to_evaluate = []
    else:
        raise NotImplementedError(f"Core {args.core} has not been implemented.")

    metric, mode = "single_trial_correlation", "max"
    hebo = HEBOSearch(
        metric=metric,
        mode=mode,
        points_to_evaluate=points_to_evaluate,
        max_concurrent=args.max_concurrent,
    )
    trainable = train_function
    if torch.cuda.device_count() > 0:
        trainable = tune.with_resources(
            train_function,
            resources={"cpu": 2, "gpu": 1},
        )
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            search_alg=hebo,
            num_samples=args.num_samples,
        ),
        run_config=RunConfig(
            local_dir=args.output_dir,
            verbose=args.verbose,
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=2),
        ),
    )

    results = tuner.fit()

    with open(os.path.join(args.output_dir, "result.pkl"), "wb") as file:
        pickle.dump(results, file)

    print(f"Best setting: {results.get_best_result().config}")


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
    parser.add_argument(
        "--core", type=str, required=True, help="The core module to use."
    )

    # training settings
    parser.add_argument(
        "--epochs", default=200, type=int, help="maximum epochs to train the model."
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="",
        help="Device to use for computation. "
        "Use the best available device if --device is not specified.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ray_address", type=int, default=6123)
    parser.add_argument("--num_samples", type=int, default=-1)

    # misc
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])

    main(parser.parse_args())
