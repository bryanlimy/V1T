import os
import wandb
import argparse
from functools import partial

import train as trainer


class Args:
    def __init__(self, id: str, config: wandb.Config, dataset: str, batch_size: int):
        self.dataset = dataset
        self.output_dir = os.path.join(config.output_dir, id)
        self.batch_size = batch_size
        self.device = ""
        self.mouse_ids = None
        self.seed = 1234
        self.save_plots = False
        self.dpi = 120
        self.format = "svg"
        self.clear_output_dir = False
        self.verbose = 0
        self.use_wandb = True
        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def main(dataset: str, batch_size: int):
    run = wandb.init(group="vit_sweep")
    config = run.config
    run.name = run.id
    run.save()

    args = Args(id=run.id, config=config, dataset=dataset, batch_size=batch_size)
    trainer.main(args, wandb_sweep=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sensorium",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--num_trials", type=int, default=1)
    params = parser.parse_args()

    wandb.agent(
        sweep_id=f"bryanlimy/sensorium/{params.sweep_id}",
        function=partial(
            main,
            dataset=params.dataset,
            batch_size=params.batch_size,
        ),
        count=params.num_trials,
    )
