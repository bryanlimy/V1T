import torch
import numpy as np
import typing as t
from tqdm import tqdm
from torch.utils.data import DataLoader


def inference(
    ds: DataLoader,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):
    """Inference data in DataLoader ds
    Returns:
        results: t.Dict[str, torch.Tensor]
            - predictions: torch.Tensor, predictions given images
            - targets: torch.Tensor, the ground-truth responses
            - mouse_id: torch.Tensor, mouse ID of the corresponding responses
            - trial_id: torch.Tensor, trial ID of the corresponding responses
            - frame_id: torch.Tensor, frame ID of the corresponding responses
    """
    results = {}
    model.train(False)
    for data in tqdm(ds, desc="Inference"):
        images = data["image"].to(device)
        predictions = model(images)
        predictions = predictions.detach().cpu()
        # separate responses by mouse ID
        for i in range(len(predictions)):
            mouse_id = int(data["mouse_id"][i])
            if mouse_id not in results:
                results[mouse_id] = {
                    "prediction": [],
                    "target": [],
                    "frame_id": [],
                    "trial_id": [],
                }
            results[mouse_id]["prediction"].append(predictions[i])
            results[mouse_id]["target"].append(data["response"][i])
            results[mouse_id]["frame_id"].append(data["frame_id"][i])
            results[mouse_id]["trial_id"].append(data["trial_id"][i])
    for mouse_id, mouse_result in results.items():
        mouse_result = {k: torch.stack(v) for k, v in mouse_result.items()}
        # crop padded neurons if exists
        if ds.dataset.padding:
            num_neurons = ds.dataset.mice_meta[mouse_id]["num_neurons"]
            mouse_result["prediction"] = mouse_result["prediction"][:, :num_neurons]
            mouse_result["target"] = mouse_result["target"][:, :num_neurons]
        results[mouse_id] = mouse_result
    return results


def correlation(
    y1: torch.Tensor,
    y2: torch.Tensor,
    dim: t.Union[None, int, t.Tuple[int]],
    eps: float = 1e-8,
):
    """Compute the correlation between y1 and y2 along dimensions dim."""
    y1 = (y1 - torch.mean(y1, dim=dim, keepdim=True)) / (
        torch.std(y1, dim=dim, keepdim=True) + eps
    )
    y2 = (y2 - torch.mean(y2, dim=dim, keepdim=True)) / (
        torch.std(y2, dim=dim, keepdim=True) + eps
    )
    return torch.mean(y1 * y2, dim=dim)


def single_trial_correlations(
    ds: DataLoader,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):
    """Compute signal trial correlation"""
    results = inference(ds=ds, model=model, device=device)
    correlations = {}
    for mouse_id, mouse_result in results.items():
        correlations[mouse_id] = correlation(
            y1=mouse_result["target"],
            y2=mouse_result["prediction"],
            dim=0,
        )
    return correlations


def average_image_correlation(
    ds: DataLoader,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):
    """Compute correlation between average responses and predictions"""
    results = inference(ds=ds, model=model, device=device)
    correlations = {}
    for mouse_id, mouse_result in results.items():
        mean_responses, mean_predictions = [], []
        # calculate mean responses and predictions with the same frame ID
        for frame_id in torch.unique(mouse_result["frame_id"]):
            indexes = torch.squeeze(torch.nonzero(mouse_result["frame_id"] == frame_id))
            responses = mouse_result["target"][indexes]
            predictions = mouse_result["prediction"][indexes]
            mean_responses.append(torch.mean(responses, dim=0, keepdim=True))
            mean_predictions.append(torch.mean(predictions, dim=0, keepdim=True))
        mean_responses = torch.vstack(mean_predictions)
        mean_predictions = torch.vstack(mean_predictions)
        correlations[mouse_id] = correlation(
            y1=mean_responses, y2=mean_predictions, dim=0
        )
    return correlations


def _feve(
    targets: t.List[torch.Tensor],
    predictions: t.List[torch.Tensor],
    threshold: float = 0.15,
):
    """Compute the fraction of explainable variance explained per neuron"""
    image_var, prediction_var = [], []
    for target, prediction in zip(targets, predictions):
        image_var.append(torch.var(target, dim=0))
        prediction_var.append((target - prediction) ** 2)
    image_var = torch.vstack(image_var)
    prediction_var = torch.vstack(prediction_var)

    total_var = torch.var(torch.vstack(targets), dim=0)
    noise_var = torch.mean(image_var, dim=0)
    fev = (total_var - noise_var) / total_var

    pred_var = torch.mean(prediction_var, dim=0)
    fev_e = 1 - (pred_var - noise_var) / (total_var - noise_var)

    # ignore neurons below FEV threshold
    fev_e = fev_e[fev >= threshold]
    return fev_e


def feve(
    ds: DataLoader,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):
    """Compute the fraction of explainable variance explained per neuron."""
    results = inference(ds=ds, model=model, device=device)
    fev_e = {}
    for mouse_id, mouse_result in results.items():
        responses, predictions = [], []
        # calculate mean responses and predictions with the same frame ID
        for frame_id in torch.unique(mouse_result["frame_id"]):
            indexes = torch.squeeze(torch.nonzero(mouse_result["frame_id"] == frame_id))
            responses.append(mouse_result["target"][indexes])
            predictions.append(mouse_result["prediction"][indexes])
        fev_e[mouse_id] = _feve(targets=responses, predictions=predictions)
    return fev_e
