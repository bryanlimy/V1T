import torch
import typing as t
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader


from v1t import losses


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
        corr = losses.correlation(y1=self.predictions, y2=self.targets, dim=0)
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
        corr = losses.correlation(y1=mean_responses, y2=mean_predictions, dim=0)
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

    @staticmethod
    def compute_oracle_corr(repeated_outputs: np.ndarray):
        if len(repeated_outputs.shape) == 3:
            _, r, n = repeated_outputs.shape
            oracles = (
                (repeated_outputs.mean(axis=1, keepdims=True) - repeated_outputs / r)
                * r
                / (r - 1)
            )
            if np.any(np.isnan(oracles)):
                print(
                    f"Warning: {np.isnan(oracles).mean() * 100}% values are NaN "
                    f"when calculating the oracle. NaNs will be set to Zero."
                )
                oracles[np.isnan(oracles)] = 0
            return losses.correlation(
                y1=oracles.reshape(-1, n), y2=repeated_outputs.reshape(-1, n), dim=0
            )
        else:
            oracles = []
            for outputs in repeated_outputs:
                r, n = outputs.shape
                # compute the mean over repeats, for each neuron
                mu = outputs.mean(axis=0, keepdims=True)
                # compute oracle predictor
                oracle = (mu - outputs / r) * r / (r - 1)

                if np.any(np.isnan(oracle)):
                    print(
                        f"Warning: {np.isnan(oracles).mean() * 100}% values are NaN "
                        f"when calculating the oracle. NaNs will be set to Zero."
                    )
                    oracle[np.isnan(oracle)] = 0

                oracles.append(oracle)
            return losses.correlation(
                y1=np.vstack(repeated_outputs), y2=np.vstack(oracles), dim=0
            )

    @staticmethod
    def compute_oracle_corr_corrected(repeated_outputs: np.ndarray):
        if len(repeated_outputs.shape) == 3:
            var_noise = repeated_outputs.var(axis=1).mean(0)
            var_mean = repeated_outputs.mean(axis=1).var(0)
        else:
            var_noise, var_mean = [], []
            for repeat in repeated_outputs:
                var_noise.append(repeat.var(axis=0))
                var_mean.append(repeat.mean(axis=0))
            var_noise = np.mean(np.array(var_noise), axis=0)
            var_mean = np.var(np.array(var_mean), axis=0)
        return var_mean / np.sqrt(var_mean * (var_mean + var_noise))

    def get_oracles(self, corrected: bool = False, per_neuron: bool = True):
        oracles = []
        repeated_targets, _ = self.split_responses()
        for repeated_target in repeated_targets:
            if corrected:
                oracle = self.compute_oracle_corr_corrected(repeated_target)
            else:
                oracle = self.compute_oracle_corr(repeated_target)
            oracles.append(oracle)
        oracles = np.hstack(oracles)
        if not per_neuron:
            oracles = np.mean(oracles)
        return oracles

    def get_fraction_oracles(self, corrected: bool = False):
        oracles = self.get_oracles(corrected=corrected, per_neuron=True)
        pred_correlations = self.single_trial_correlation(per_neuron=True)
        oracle_performance, _, _, _ = np.linalg.lstsq(
            np.hstack(oracles)[:, np.newaxis], np.hstack(pred_correlations)
        )
        return oracle_performance
