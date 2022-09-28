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
    args,
    ds: t.Dict[int, DataLoader],
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
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
    for mouse_id, dataloader in tqdm(
        ds.items(), desc="Evaluation", disable=args.verbose == 0
    ):
        result = {
            "images": [],
            "predictions": [],
            "targets": [],
            "trial_ids": [],
            "frame_ids": [],
        }
        for batch in dataloader:
            images = batch["image"].to(device)
            predictions = model(images, mouse_id)
            result["predictions"].append(
                dataloader.dataset.i_transform_response(predictions.detach().cpu())
            )
            result["targets"].append(
                dataloader.dataset.i_transform_response(batch["response"])
            )
            images = dataloader.dataset.i_transform_image(images.detach().cpu())
            result["images"].append(images)
            result["frame_ids"].append(batch["frame_id"])
            result["trial_ids"].append(batch["trial_id"])
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
    eval_result = {}
    outputs = inference(args, ds=ds, model=model, device=args.device)
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


def set_device(args):
    """Set args.device to a torch.device"""
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
