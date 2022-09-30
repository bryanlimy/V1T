import os
import copy
import torch
import random
import subprocess
import numpy as np
import typing as t
import pandas as pd
from torch import nn
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader

from sensorium import metrics
from sensorium.utils import yaml, tensorboard


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for Python, Numpy asn PyTorch.
    Args:
        seed: int, the random seed to use.
        deterministic: bool, use "deterministic" algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def inference(
    args, ds: t.Dict[int, DataLoader], model: torch.nn.Module
) -> t.Dict[int, t.Dict[str, torch.Tensor]]:
    """Inference data in DataLoader ds
    Returns:
        results: t.Dict[int, t.Dict[str, torch.Tensor]]
            - mouse_id
                - predictions: torch.Tensor, predictions given images
                - targets: torch.Tensor, the ground-truth responses
                - images: torch.Tensor, the natural images
                - trial_ids: torch.Tensor, trial ID of the responses
                - frame_ids: torch.Tensor, frame ID of the responses
    """
    results = {}
    model.train(False)
    for mouse_id, mouse_ds in tqdm(
        ds.items(), desc="Evaluation", disable=args.verbose == 0
    ):
        if mouse_id in (0, 1) and mouse_ds.dataset.tier == "test":
            # skip Mouse 1 and 2 on test set as target responses are not provided
            continue
        result = {
            "images": [],
            "predictions": [],
            "targets": [],
            "trial_ids": [],
            "frame_ids": [],
        }
        for data in mouse_ds:
            predictions = model(data["image"].to(model.device), mouse_id=mouse_id)
            predictions = predictions.detach().cpu()
            result["predictions"].append(
                mouse_ds.dataset.i_transform_response(predictions)
            )
            result["targets"].append(
                mouse_ds.dataset.i_transform_response(data["response"])
            )
            images = mouse_ds.dataset.i_transform_image(data["image"])
            result["images"].append(images)
            result["frame_ids"].append(data["frame_id"])
            result["trial_ids"].append(data["trial_id"])
        results[mouse_id] = {k: torch.cat(v, dim=0) for k, v in result.items()}
    return results


def evaluate(
    args,
    ds: t.Dict[int, DataLoader],
    model: nn.Module,
    epoch: int,
    summary: tensorboard.Summary,
    mode: int = 1,
):
    """Evaluate DataLoaders ds on the 3 challenge metrics"""
    eval_result = {}
    outputs = inference(args, ds=ds, model=model)
    trial_correlations = metrics.single_trial_correlations(results=outputs)
    summary.plot_correlation(
        "metrics/single_trial_correlation",
        data=metrics2df(trial_correlations),
        step=epoch,
        mode=mode,
    )
    eval_result["trial_correlation"] = {
        mouse_id: torch.mean(correlation)
        for mouse_id, correlation in trial_correlations.items()
    }
    if mode == 2:  # only test set has repeated images
        image_correlations = metrics.average_image_correlation(results=outputs)
        summary.plot_correlation(
            "metrics/average_image_correlation",
            data=metrics2df(image_correlations),
            step=epoch,
            mode=mode,
        )
        eval_result["image_correlation"] = {
            mouse_id: torch.mean(correlation)
            for mouse_id, correlation in image_correlations.items()
        }
        feve = metrics.feve(results=outputs)
        summary.plot_correlation(
            "metrics/FEVE",
            data=metrics2df(feve),
            step=epoch,
            ylabel="FEVE",
            mode=mode,
        )
        eval_result["feve"] = {
            mouse_id: torch.mean(f_eve) for mouse_id, f_eve in feve.items()
        }
    # write individual and average results to TensorBoard
    for metric, results in eval_result.items():
        for mouse_id, result in results.items():
            summary.scalar(
                tag=f"{metric}/mouse{mouse_id}",
                value=result,
                step=epoch,
                mode=mode,
            )
        summary.scalar(
            tag=f"{metric}/average",
            value=np.mean(list(results.values())),
            step=epoch,
            mode=mode,
        )
    # plot image and response pairs
    summary.plot_image_response(
        tag=f"image_response", results=outputs, step=epoch, mode=mode
    )
    return eval_result


def update_dict(target: dict, source: dict, replace: bool = False):
    """Update target dictionary with values from source dictionary"""
    for k, v in source.items():
        if replace:
            target[k] = v
        else:
            if k not in target:
                target[k] = []
            target[k].append(v)


def check_output(command: list):
    """Run command in subprocess and return output as string"""
    return subprocess.check_output(command).strip().decode()


def save_args(args):
    """Save args object as dictionary to args.output_dir/args.json"""
    arguments = copy.deepcopy(args.__dict__)
    arguments["git_hash"] = check_output(["git", "describe", "--always"])
    arguments["hostname"] = check_output(["hostname"])
    yaml.save(filename=os.path.join(args.output_dir, "args.yaml"), data=arguments)


def load_args(args):
    """Load args object from args.output_dir/args.yaml"""
    content = yaml.load(os.path.join(args.output_dir, "args.yaml"))
    for key, value in content.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def get_device(args):
    """Get the appropriate torch.device from args.device argument"""
    device = args.device
    if not device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    args.device = torch.device(device)


def metrics2df(results: t.Dict[str, torch.Tensor]):
    mouse_ids, values = [], []
    for mouse_id, v in results.items():
        mouse_ids.extend([mouse_id] * len(v))
        values.extend(v.tolist())
    return pd.DataFrame({"mouse": mouse_ids, "results": values})


def log_metrics(
    results: t.Union[
        t.Dict[str, t.List[torch.Tensor]],
        t.Dict[int, t.Dict[str, torch.Tensor]],
    ],
    epoch: int,
    mode: int,
    summary: tensorboard.Summary,
    mouse_id: int = None,
):
    """Compute the mean of the metrics in results and log to Summary

    Args:
        results: t.Union[
                t.Dict[str, t.List[torch.Tensor]],
                t.Dict[int, t.Dict[str, torch.Tensor]]
            ]: a dictionary of tensors where keys are the name of the metrics
            that represent results from of a mouse, or a dictionary of a
            dictionary of tensors where the keys are the mouse IDs that
            represents the average results of multiple mice.
            When mouse_id is provided, it assumes the former.
        epoch: int, the current epoch number.
        mode: int, Summary logging mode.
        summary: tensorboard.Summary, Summary class
        mouse_id: int, the mouse_id of the result dictionary, None if the
            dictionary represents results from multiple mice.
    """
    if mouse_id is not None:
        for metric, values in results.items():
            results[metric] = torch.stack(values).mean()
            summary.scalar(
                f"{metric}/mouse{mouse_id}",
                value=results[metric],
                step=epoch,
                mode=mode,
            )
    else:
        mouse_ids = list(results.keys())
        metrics = list(results[mouse_ids[0]].keys())
        for metric in metrics:
            results[metric] = torch.stack(
                [results[mouse_id][metric] for mouse_id in mouse_ids]
            ).mean()
            summary.scalar(metric, value=results[metric], step=epoch, mode=mode)
