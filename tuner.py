import os
import ray
import sys
import torch
import pickle
import argparse
from ray import tune
from shutil import rmtree
from os.path import abspath
from ray.air import session
from functools import partial
from datetime import datetime
from ray.tune import CLIReporter
from ray.tune.search.hebo import HEBOSearch
from ray.tune.schedulers import ASHAScheduler

import train as trainer


class Logger:
    def __init__(self, filename: str):
        self.console = sys.stdout
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.file = open(filename, "w")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


def get_timestamp():
    return f"{datetime.now():%Y%m%d-%Hh%Mm}"


def trial_name_creator(trial: tune.experiment.trial.Trial):
    return f"{trial.trial_id}"


def trial_dirname_creator(trial: tune.experiment.trial.Trial):
    return f"{get_timestamp()}-{trial.trial_id}"


class Args:
    def __init__(
        self,
        config: dict,
        data_dir: str,
        output_dir: str,
        readout: str,
        epochs: int,
        batch_size: int,
        num_workers: int,
        device: str,
    ):
        self.dataset = data_dir
        self.output_dir = output_dir
        self.readout = readout
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.mouse_ids = None
        self.pretrain_core = ""
        self.depth_scale = 1
        self.seed = 1234
        self.save_plots = False
        self.dpi = 120
        self.format = "svg"
        self.clear_output_dir = False
        self.verbose = 0
        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def train_function(
    config: dict,
    data_dir: str,
    output_dir: str,
    readout: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
):
    args = Args(
        config,
        data_dir=data_dir,
        output_dir=os.path.join(output_dir, session.get_trial_id()),
        readout=readout,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    return trainer.main(args)


def get_search_space(args):
    # default search space
    search_space = {
        "plus": False,
        "disable_grid_predictor": tune.choice([True, False]),
        "grid_predictor_dim": tune.choice([2, 3]),
        "bias_mode": tune.choice([0, 1, 2]),
        "criterion": tune.choice(["rmsse", "poisson", "correlation"]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "core_reg_scale": tune.loguniform(1e-5, 1),
        "readout_reg_scale": tune.loguniform(1e-5, 1),
        "shifter_reg_scale": tune.loguniform(1e-5, 1),
        "ds_scale": tune.choice([True, False]),
        "crop_mode": tune.choice([1, 2]),
        "core_lr_scale": tune.uniform(0, 1),
        "use_shifter": tune.choice([True, False]),
    }

    if args.core == "vit":
        search_space.update(
            {
                "core": "vit",
                "dropout": tune.uniform(0, 0.8),
                "patch_size": tune.randint(1, 10),
                "emb_dim": tune.randint(8, 128),
                "num_heads": tune.randint(1, 16),
                "mlp_dim": tune.randint(8, 128),
                "num_layers": tune.randint(1, 8),
                "dim_head": tune.randint(8, 128),
            }
        )
        points_to_evaluate = [
            {
                "dropout": 0.25,
                "patch_size": 4,
                "emb_dim": 64,
                "num_heads": 3,
                "mlp_dim": 64,
                "num_layers": 3,
                "dim_head": 64,
                "disable_grid_predictor": False,
                "grid_predictor_dim": 2,
                "bias_mode": 0,
                "criterion": "poisson",
                "lr": 1e-3,
                "core_reg_scale": 0,
                "readout_reg_scale": 0.0076,
                "shifter_reg_scale": 0,
                "ds_scale": True,
                "crop_mode": 1,
                "core_lr_scale": 1,
                "use_shifter": False,
            }
        ]
        evaluated_rewards = [0.29188114404678345]
    elif args.core == "stacked2d":
        search_space.update(
            {
                "core": "stacked2d",
                "num_layers": tune.randint(1, 8),
                "dropout": tune.uniform(0, 0.8),
            }
        )
        points_to_evaluate = [
            {
                "num_layers": 4,
                "dropout": 0.0,
                "disable_grid_predictor": False,
                "grid_predictor_dim": 2,
                "bias_mode": 0,
                "criterion": "poisson",
                "lr": 1e-3,
                "core_reg_scale": 1,
                "readout_reg_scale": 0.0076,
                "shifter_reg_scale": 0,
                "ds_scale": True,
                "crop_mode": 1,
                "core_lr_scale": 1,
                "use_shifter": False,
            }
        ]
        evaluated_rewards = [0.30914896726608276]
    elif args.core == "stn":
        search_space.update(
            {
                "core": "stn",
                "num_filters": tune.randint(8, 64),
                "num_layers": tune.randint(1, 8),
                "dropout": tune.uniform(0, 0.8),
            }
        )
        points_to_evaluate = []
        evaluated_rewards = []
    else:
        raise NotImplementedError(f"Core {args.core} has not been implemented.")

    return search_space, points_to_evaluate, evaluated_rewards


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    search_space, points_to_evaluate, evaluated_rewards = get_search_space(args)

    metric, mode = "single_trial_correlation", "max"
    num_gpus = torch.cuda.device_count()
    max_concurrent = max(1, num_gpus)

    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=args.epochs // 10,
        grace_period=2,
        reduction_factor=2,
    )
    hebo = HEBOSearch(
        metric=metric,
        mode=mode,
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        max_concurrent=max_concurrent,
    )
    reporter = CLIReporter(
        metric_columns=[
            "iterations",
            "single_trial_correlation",
            "correlation_to_average",
        ],
        parameter_columns=[
            "criterion",
            "num_layers",
            "dropout",
            "bias_mode",
            "crop_mode",
            "use_shifter",
            "disable_grid_predictor",
        ],
        max_progress_rows=10,
        max_error_rows=3,
        max_column_length=20,
        max_report_frequency=60,
        metric=metric,
        mode=mode,
        sort_by_metric=True,
    )

    name = (
        os.path.basename(args.resume_dir)
        if args.resume_dir
        else f"{get_timestamp()}-{args.core}"
    )

    sys.stdout = Logger(filename=os.path.join(args.output_dir, name, "output.log"))

    results = tune.run(
        partial(
            train_function,
            data_dir=abspath(args.dataset),
            output_dir=abspath(os.path.join(args.output_dir, name, "output_dir")),
            readout="gaussian2d",
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        ),
        name=name,
        resources_per_trial={"cpu": args.num_cpus, "gpu": 1 if num_gpus else 0},
        config=search_space,
        num_samples=args.num_samples,
        local_dir=args.output_dir,
        search_alg=hebo,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_score_attr="single_trial_correlation",
        checkpoint_freq=1,
        checkpoint_at_end=True,
        verbose=args.verbose,
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_dirname_creator,
        resume="LOCAL" if args.resume_dir else None,
        max_concurrent_trials=max_concurrent,
    )

    with open(os.path.join(args.output_dir, "result.pkl"), "wb") as file:
        pickle.dump(results.get_best_result(), file)

    print(f"\n\nBest setting\n{results.get_best_result()}")


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
    parser.add_argument("--resume_dir", type=str, default="")
    parser.add_argument("--num_cpus", type=int, default=3)
    parser.add_argument(
        "--num_workers", default=2, type=int, help="number of works for DataLoader."
    )
    parser.add_argument("--plus", actiion="store_true")

    # model settings
    parser.add_argument(
        "--core", type=str, required=True, help="The core module to use."
    )
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
