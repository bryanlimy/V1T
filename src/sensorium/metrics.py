import torch
import typing as t


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


def single_trial_correlations(results: t.Dict[int, t.Dict[str, torch.Tensor]]):
    """Compute signal trial correlation"""
    correlations = {}
    for mouse_id, mouse_result in results.items():
        correlations[mouse_id] = correlation(
            y1=mouse_result["targets"],
            y2=mouse_result["predictions"],
            dim=0,
        )
    return correlations


def average_image_correlation(results: t.Dict[int, t.Dict[str, torch.Tensor]]):
    """Compute correlation between average responses and predictions"""
    correlations = {}
    for mouse_id, mouse_result in results.items():
        mean_responses, mean_predictions = [], []
        # calculate mean responses and predictions with the same frame ID
        for frame_id in torch.unique(mouse_result["frame_ids"]):
            indexes = torch.where(mouse_result["frame_ids"] == frame_id)[0]
            responses = mouse_result["targets"][indexes]
            predictions = mouse_result["predictions"][indexes]
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


def feve(results: t.Dict[int, t.Dict[str, torch.Tensor]]):
    """Compute the fraction of explainable variance explained per neuron."""
    fev_e = {}
    for mouse_id, mouse_result in results.items():
        responses, predictions = [], []
        # calculate mean responses and predictions with the same frame ID
        for frame_id in torch.unique(mouse_result["frame_ids"]):
            indexes = torch.where(mouse_result["frame_ids"] == frame_id)[0]
            responses.append(mouse_result["targets"][indexes])
            predictions.append(mouse_result["predictions"][indexes])
        fev_e[mouse_id] = _feve(targets=responses, predictions=predictions)
    return fev_e
