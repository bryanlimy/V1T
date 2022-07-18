import torch
import numpy as np
import typing as t
from tqdm import tqdm


def inference(ds, model, device: torch.device = torch.device("cpu")):
    predictions, targets = {}, {}
    model.train(False)
    for data in tqdm(ds, desc="Inference"):
        images = data["image"].to(device)
        responses = model(images)
        responses = responses.detach().cpu()
        for i, mouse_id in enumerate(data["mouse_id"].tolist()):
            if mouse_id not in predictions:
                predictions[mouse_id] = []
            if mouse_id not in targets:
                targets[mouse_id] = []
            predictions[mouse_id].append(responses[i])
            targets[mouse_id].append(data["response"][i])
    predictions = {k: torch.stack(v) for k, v in predictions.items()}
    targets = {k: torch.stack(v) for k, v in targets.items()}
    return predictions, targets


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


def single_trial_correlations(ds, model, device: torch.device = torch.device("cpu")):
    predictions, targets = inference(ds=ds, model=model, device=device)
    correlations = {}
    for mouse_id in predictions.keys():
        correlations[mouse_id] = correlation(
            y1=targets[mouse_id], y2=predictions[mouse_id], dim=0
        )
        if torch.any(np.isnan(correlations[mouse_id])):
            print(
                f"{torch.isnan(correlations[mouse_id]).mean() * 100} NaN values, set to zeros."
            )
        correlations[mouse_id] = torch.nan_to_num(correlations[mouse_id])
    return correlations
