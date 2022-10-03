import os
import torch
import typing as t
import numpy as np
from torch import nn
from sensorium.models.model import Model


def load_pretrain_core(args, model: Model):
    filename = os.path.join(args.pretrain_core, "ckpt", "best_model.pt")
    assert os.path.exists(filename), f"Cannot find pretrain core {filename}."
    model_dict = model.state_dict()
    checkpoint = torch.load(filename, map_location=model.device)
    pretrain_dict = checkpoint["model_state_dict"]
    # load core parameters in pretrain_dict which should be the same as model_dict
    core_dict = {}
    try:
        for name, parameter in model_dict.items():
            if name.startswith("core"):
                core_dict[name] = pretrain_dict[name]
    except KeyError as e:
        raise KeyError(f"Pretrained core module contains different parameters: {e}")
    model_dict.update(core_dict)
    model.load_state_dict(model_dict)
    if args.verbose:
        print(f"\nLoaded pretrained core from {args.pretrain_core}.\n")


class Checkpoint:
    """Checkpoint class to save and load checkpoints and early stopping.

    The monitor function monitors the given loss value and compare it against
    the previous best loss recorded. After min_epochs number of epochs, if the
    current loss value has not improved for more than patience number of epoch,
    then send terminate flag and load the best weight for the model.
    """

    def __init__(
        self,
        args,
        model: nn.Module,
        optimizer: torch.optim = None,
        scheduler: torch.optim.lr_scheduler = None,
        patience: int = 10,
        min_epochs: int = 50,
    ):
        """
        Args:
            args: argparse parameters.
            model: nn.Module, model.
            optimizer: torch.optim, optimizer.
            scheduler: torch.optim.lr_scheduler, scheduler.
            patience: int, the number of epochs to wait until terminate if the
                loss value does not improve.
            min_epochs: int, number of epochs to train the model before early
                stopping begins monitoring.
        """
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.patience = patience
        self.min_epochs = min_epochs

        self.checkpoint_dir = os.path.join(args.output_dir, "ckpt")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self._wait = 0
        self.best_loss = np.inf
        self.best_epoch = -1
        self.best_weights = None
        self._verbose = args.verbose
        self._device = args.device

    def save(self, loss: t.Union[float, np.ndarray, torch.Tensor], epoch: int):
        """Save current model as best_model.pt"""
        filename = os.path.join(self.checkpoint_dir, "best_model.pt")
        checkpoint = {
            "epoch": epoch,
            "loss": float(loss),
            "model_state_dict": self._model.state_dict(),
        }
        if self._optimizer is not None:
            checkpoint["optimizer_state_dict"] = self._optimizer.state_dict()
        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()
        torch.save(checkpoint, f=filename)
        if self._verbose:
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
            checkpoint = torch.load(filename, map_location=self._device)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            if self._optimizer is not None and "optimizer_state_dict" in checkpoint:
                self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self._scheduler is not None and "scheduler_state_dict" in checkpoint:
                self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint["epoch"]
            if self._verbose:
                print(f"\nLoaded checkpoint (epoch {epoch}) from {filename}.\n")
        elif force:
            raise FileNotFoundError(f"Cannot find checkpoint in {self.checkpoint_dir}.")
        return epoch

    def monitor(self, loss: t.Union[float, torch.Tensor], epoch: int) -> bool:
        terminate = False
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self._wait = 0
            self.save(epoch=epoch, loss=loss)
        elif epoch > self.min_epochs:
            if self._wait < self.patience:
                self._wait += 1
            else:
                terminate = True
                if self._verbose:
                    print(f"Model has not improved in {self._wait} epochs.")
        return terminate
