import os
import torch
import numpy as np
from torch import nn
from glob import glob
import typing as t
from sensorium.utils import yaml


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
        optimizer: torch.optim,
        patience: int = 10,
        min_epochs: int = 50,
    ):
        """
        Args:
            args: argparse parameters.
            model: nn.Module, model.
            optimizer: torch.optim, optimizer.
            patience: int, the number of epochs to wait until terminate if the
                loss value does not improve.
            min_epochs: int, number of epochs to train the model before early
                stopping begins monitoring.
        """
        self._model = model
        self._optimizer = optimizer
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
        torch.save(
            {
                "epoch": epoch,
                "loss": float(loss),
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            f=filename,
        )
        if self._verbose:
            print(f"\n Checkpoint saved to {filename}.\n")

    def restore(self) -> int:
        """Load the best model in self.checkpoint_dir and return the epoch"""
        epoch = 0
        filename = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self._device)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            if self._verbose:
                print(f"\nLoaded checkpoint from {filename}.\n")
        return epoch

    def monitor(self, loss: float, epoch: int) -> bool:
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
                self.restore()
                if self._verbose:
                    print(f"model has not improved in {self._wait} epochs.\n")
        return terminate
