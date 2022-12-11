import os
import wandb

import train as trainer


class Args:
    def __init__(self, id: str, config: wandb.Config):
        self.dataset = config.dataset
        self.output_dir = os.path.join(config.output_dir, id)
        self.readout = config.readout
        self.epochs = config.epochs
        self.batch_size = config.batch_size
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


def main():
    run = wandb.init(group="vit_sweep")
    config = run.config
    run.name = run.id
    run.save()

    args = Args(id=run.id, config=config)
    trainer.main(args, wandb_sweep=True)


main()
