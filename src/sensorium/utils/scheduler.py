import torch
from torch.optim import Optimizer, lr_scheduler
import typing as t
import numpy as np
import os
from sensorium.models import Model


class Scheduler:
    def __init__(
        self,
        args,
        mode: t.Literal["min", "max"],
        model: Model,
        optimizer: Optimizer,
        max_reduce: int = 2,
        min_lr: float = 1e-6,
        lr_patience: int = 5,
        factor: float = 0.5,
        min_epochs: int = 50,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ):
        """
        Args:
            args: argparse parameters.
            mode: 'min' or 'max', compare objective by minimum or maximum
            model: Model, model.
            optimizer: torch.optim, optimizer.
            max_reduce: int, maximum number of learning rate reductions before
                terminating early stopping.
            min_lr: float, minimum learning rate.
            lr_patience: int, the number of epochs to wait before reducing the
                learning rate.
            factor: float, learning rate reduction factor.
                i.e. new_lr = max(factor * old_lr, min_lr)
            min_epochs: int, number of epochs to train the model before early
                stopping begins monitoring.
            save_optimizer: bool, save optimizer state dict to checkpoint if True.
        """
        assert mode in ("min", "max")
        self.mode = mode
        self.model = model
        self.optimizer = optimizer
        self.max_reduce = max_reduce
        self.num_reduce = 0
        self.lr_patience = lr_patience
        self.lr_wait = 0
        self.min_lr = min_lr
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor
        self.min_epochs = min_epochs
        self.best_value = torch.inf if mode == "min" else -torch.inf

        self.checkpoint_dir = os.path.join(args.output_dir, "ckpt")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.device = args.device
        self.verbose = args.verbose

    def save_checkpoint(
        self, value: t.Union[float, np.ndarray, torch.Tensor], epoch: int
    ):
        """Save current model as best_model.pt"""
        filename = os.path.join(self.checkpoint_dir, "best_model.pt")
        ckpt = {
            "epoch": epoch,
            "value": float(value),
            "model_state_dict": self.model.state_dict(),
        }
        if self.save_optimizer:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.save_scheduler:
            ckpt["scheduler_state_dict"] = self.state_dict()
        torch.save(ckpt, f=filename)
        print(f"\nCheckpoint saved to {filename}.")

    def restore(self, force: bool = False) -> int:
        """
        Load the best model in self.checkpoint_dir and return the epoch
        Args:
            force: bool, raise an error if checkpoint is not found.
        Return:
            epoch: int, the number of epoch the model has been trained for,
                return 0 if no checkpoint was found.
        """
        epoch = 0
        filename = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location=self.device)
            epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.load_state_dict(ckpt["scheduler_state_dict"])
            if self.verbose:
                print(f"\nLoaded checkpoint (epoch {epoch}) from {filename}.\n")
        elif force:
            raise FileNotFoundError(f"Cannot find checkpoint in {self.checkpoint_dir}.")
        return epoch

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "model")
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def is_better(self, value: t.Union[float, torch.Tensor]):
        if self.mode == "min":
            return value < self.best_value
        else:
            return value > self.best_value

    def reduce_lr(self):
        """Reduce the learning rates for each param_group by the defined factor"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group["lr"] = new_lr
            if self.verbose == 2:
                print(f'Reduce learning rate of {param_group["name"]} to {new_lr}.')

    def step(self, value: t.Union[float, torch.Tensor], epoch: int):
        terminate = False
        if self.is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            self.lr_wait = 0
            self.num_reduce = 0
            self.save_checkpoint(value=value, epoch=epoch)
        elif epoch > self.min_epochs:
            if self.lr_wait >= self.lr_patience:
                if self.num_reduce >= self.max_reduce:
                    terminate = True
                    print(
                        f"Model has not improved after {self.num_reduce} LR reductions."
                    )
                else:
                    self.reduce_lr()
                    self.num_reduce += 1
                    self.lr_wait = 0
            else:
                self.lr_wait += 1
        return terminate
