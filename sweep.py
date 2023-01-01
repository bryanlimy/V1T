import os
import wandb
import argparse
from functools import partial
from datetime import datetime

import train as trainer


class Args:
    def __init__(
        self,
        id: str,
        config: wandb.Config,
        dataset: str,
        num_workers: int = 2,
        verbose: int = 1,
    ):
        self.dataset = dataset
        self.output_dir = os.path.join(
            config.output_dir, f"{datetime.now():%Y%m%d-%Hh%Mm}-{id}"
        )
        self.num_workers = num_workers
        self.device = ""
        self.mouse_ids = None
        self.seed = 1234
        self.save_plots = False
        self.dpi = 120
        self.format = "svg"
        self.clear_output_dir = False
        self.amp = False
        self.verbose = verbose
        self.use_wandb = True
        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def main(wandb_group: str, dataset: str, num_workers: int = 2):
    run = wandb.init(group=wandb_group)
    config = run.config
    run.name = run.id
    args = Args(
        id=run.id,
        config=config,
        dataset=dataset,
        num_workers=num_workers,
    )
    trainer.main(args, wandb_sweep=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sensorium",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--wandb_group", type=str, required=True)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    params = parser.parse_args()

    wandb.agent(
        sweep_id=f"bryanlimy/sensorium/{params.sweep_id}",
        function=partial(
            main,
            wandb_group=params.wandb_group,
            dataset=params.dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
        ),
        count=params.num_trials,
    )
