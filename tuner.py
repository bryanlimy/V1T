import os
import ray
import torch
import pickle
import argparse
from shutil import rmtree
from ray import air, tune
from ray.air import session
from ray.air.config import RunConfig
from ray.tune.search.hebo import HEBOSearch
from os.path import abspath
from ray.tune.schedulers import ASHAScheduler

import train as trainer


class Args:
    def __init__(self, config, trial_id: str):
        self.output_dir = os.path.join(config["output_dir"], trial_id)
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


def train_function(config):
    args = Args(config, trial_id=session.get_trial_id())
    return trainer.main(args)


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
        "num_workers": args.num_workers,
        "plus": False,
        "device": args.device,
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

    if args.resume_dir:
        tuner = tune.Tuner.restore(abspath(args.resume_dir))
        tuner._local_tuner._is_restored = True
        tuner._local_tuner._param_space = search_space
        # tuner._local_tuner._tune_config.num_samples = args.num_samples
        # del tuner._local_tuner._resume_config
    else:
        metric, mode = "single_trial_correlation", "max"
        num_gpus = torch.cuda.device_count()
        max_concurrent = max(1, num_gpus)

        hebo = HEBOSearch(
            metric=metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            evaluated_rewards=evaluated_rewards,
            max_concurrent=max_concurrent,
        )
        scheduler = ASHAScheduler(
            time_attr="iterations",
            max_t=args.epochs // 10,
            grace_period=2,
            reduction_factor=2,
        )

        trainable = train_function
        if num_gpus > 0:
            trainable = tune.with_resources(
                train_function,
                resources={"cpu": args.num_cpus, "gpu": 1},
            )
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                search_alg=hebo,
                scheduler=scheduler,
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
