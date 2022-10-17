import torch
import typing as t
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy


def correlation(
    y1: t.Union[torch.Tensor, np.ndarray],
    y2: t.Union[torch.Tensor, np.ndarray],
    axis: t.Union[None, int, t.Tuple[int]] = -1,
    eps: float = 1e-8,
    **kwargs
):
    """
    Compute the correlation between two NumPy arrays along the specified dimension(s).
    """
    is_tensor = isinstance(y1, torch.Tensor)
    if is_tensor:
        y1, y2 = y1.cpu().numpy(), y2.cpu().numpy()
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (
        y1.std(axis=axis, keepdims=True, ddof=0) + eps
    )
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (
        y2.std(axis=axis, keepdims=True, ddof=0) + eps
    )
    corr = (y1 * y2).mean(axis=axis, **kwargs)
    return torch.tensor(corr, dtype=torch.float32) if is_tensor else corr


class Metrics:
    def __init__(self, ds: DataLoader, results: t.Dict[str, torch.Tensor]):
        """
        Computes performance metrics of neural response predictions.
        """
        self.repeat_image = ds.dataset.tier == "test"
        self.hashed = ds.dataset.hashed
        self.targets = results["targets"].numpy()
        self.predictions = results["predictions"].numpy()
        self.image_ids = results["image_ids"].numpy()
        self.neuron_ids = deepcopy(ds.dataset.neuron_ids)
        self.trial_ids = results["trial_ids"]
        if not self.hashed:
            self.trial_ids = self.trial_ids.numpy()
            self.order()

    def order(self):
        """Re-order the responses based on trial IDs and neuron IDs."""
        trial_ids = np.argsort(self.trial_ids)
        neuron_ids = np.argsort(self.neuron_ids)

        self.targets = self.targets[trial_ids, :][:, neuron_ids]
        self.predictions = self.predictions[trial_ids, :][:, neuron_ids]
        self.image_ids = self.image_ids[trial_ids]
        self.neuron_ids = self.neuron_ids[neuron_ids]
        self.trial_ids = trial_ids

    def split_responses(
        self,
    ) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray]]:
        """
        Split the responses (or predictions) array based on image ids.
        Each element of the list contains the responses to repeated
        presentations of a single image.
        Returns:
            targets: t.List[np.ndarray]: a list of array where each tensor
                is the target responses from repeated images.
            predictions: t.List[np.ndarray]: a list of array where each tensor
                is the predicted responses from repeated images.
        """
        repeat_targets, repeat_predictions = [], []
        for image_id in np.unique(self.image_ids):
            indexes = self.image_ids == image_id
            repeat_targets.append(self.targets[indexes])
            repeat_predictions.append(self.predictions[indexes])
        return repeat_targets, repeat_predictions

    def single_trial_correlation(self, per_neuron: bool = False):
        """
        Compute single-trial correlation.
        Returns:
            corr: t.Union[float, np.ndarray], single trial correlation
        """
        corr = correlation(y1=self.predictions, y2=self.targets, axis=0)
        return corr if per_neuron else corr.mean()

    def correlation_to_average(self, per_neuron: bool = False):
        """
        Compute correlation to average response across repeats.
        Returns:
            np.array or float: Correlation (average across repeats) between responses and predictions
        """
        if not self.repeat_image or self.hashed:
            return None
        mean_responses, mean_predictions = [], []
        for repeat_responses, repeat_predictions in zip(*self.split_responses()):
            mean_responses.append(repeat_responses.mean(axis=0, keepdims=True))
            mean_predictions.append(repeat_predictions.mean(axis=0, keepdims=True))
        mean_responses = np.vstack(mean_responses)
        mean_predictions = np.vstack(mean_predictions)
        corr = correlation(y1=mean_responses, y2=mean_predictions, axis=0)
        return corr if per_neuron else corr.mean()

    def _fev(
        self,
        targets: t.List[np.ndarray],
        predictions: t.List[np.ndarray],
        return_exp_var: bool = False,
    ):
        """
        Compute the fraction of explainable variance explained per neuron
        Args:
            targets (array-like): Neuronal neuron responses (ground truth) to image repeats. Dimensions:
                [num_images] np.array(num_repeats, num_neurons)
            outputs (array-like): Model predictions to the repeated images, with an identical shape as the targets
            return_exp_var (bool): returns the fraction of explainable variance per neuron if set to True
        Returns:
            FEVe (np.array): the fraction of explainable variance explained per neuron
            --- optional: FEV (np.array): the fraction
        """
        img_var = []
        pred_var = []
        for target, prediction in zip(targets, predictions):
            pred_var.append((target - prediction) ** 2)
            img_var.append(np.var(target, axis=0, ddof=1))
        pred_var = np.vstack(pred_var)
        img_var = np.vstack(img_var)

        total_var = np.var(np.vstack(targets), axis=0, ddof=1)
        noise_var = np.mean(img_var, axis=0)
        fev = (total_var - noise_var) / total_var

        pred_var = np.mean(pred_var, axis=0)
        fev_e = 1 - (pred_var - noise_var) / (total_var - noise_var)
        return [fev, fev_e] if return_exp_var else fev_e

    def feve(self, per_neuron: bool = False, fev_threshold: float = 0.15):
        """
        Compute fraction of explainable variance explained
        Returns:
            fevl_val: t.Union[float, np.ndarray], FEVE value
        """
        if not self.repeat_image or self.hashed:
            return None
        repeat_targets, repeat_predictions = self.split_responses()
        fev_val, feve_val = self._fev(
            targets=repeat_targets,
            predictions=repeat_predictions,
            return_exp_var=True,
        )
        # ignore neurons below FEV threshold
        feve_val = feve_val[fev_val >= fev_threshold]
        return feve_val if per_neuron else feve_val.mean()
