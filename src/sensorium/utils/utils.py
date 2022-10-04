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
from torch.utils.data import DataLoader

from sensorium.metrics import Metrics
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


def inference(ds: DataLoader, model: torch.nn.Module) -> t.Dict[str, torch.Tensor]:
    """Inference data in DataLoader ds
    Returns:
        results: t.Dict[int, t.Dict[str, torch.Tensor]]
            - mouse_id
                - predictions: torch.Tensor, predictions given images
                - targets: torch.Tensor, the ground-truth responses
                - images: torch.Tensor, the natural images
                - trial_ids: torch.Tensor, trial ID of the responses
                - image_ids: torch.Tensor, image ID of the responses
    """
    results = {
        "images": [],
        "predictions": [],
        "targets": [],
        "trial_ids": [],
        "image_ids": [],
    }
    model.train(False)
    with torch.no_grad():
        for data in ds:
            predictions = model(
                data["image"].to(model.device),
                mouse_id=ds.dataset.mouse_id,
            )
            results["predictions"].append(predictions.cpu())
            results["targets"].append(data["response"])
            results["images"].append(ds.dataset.i_transform_image(data["image"]))
            results["image_ids"].append(data["image_id"])
            results["trial_ids"].append(data["trial_id"])
    results = {
        k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
        for k, v in results.items()
    }
    return results


def evaluate(
    args,
    ds: t.Dict[int, DataLoader],
    model: nn.Module,
    epoch: int = 0,
    summary: tensorboard.Summary = None,
    mode: int = 1,
    print_result: bool = False,
    save_result: str = None,
):
    """
    Evaluate DataLoaders ds on the 3 challenge metrics

    Args:
        args
        ds: t.Dict[int, DataLoader], dictionary of DataLoader, one for each mouse.
        model: nn.Module, the model.
        epoch: int, the current epoch number.
        summary: tensorboard.Summary (optional), log result to TensorBoard.
        mode: int (optional), Summary mode.
        print_result: bool, print result if True.
        save_result: str, path where the result is saved if provided.
    """
    results = {"trial_correlation": {}, "image_correlation": {}, "feve": {}}
    trial_corrs, image_corrs, feves = {}, {}, {}
    for mouse_id, mouse_ds in tqdm(
        ds.items(), desc="Evaluation", disable=args.verbose == 0
    ):
        if mouse_id in (0, 1) and mouse_ds.dataset.tier == "test":
            continue
        mouse_result = inference(ds=mouse_ds, model=model)
        if summary is not None:
            summary.plot_image_response(
                tag=f"image_response/mouse{mouse_id}",
                results=mouse_result,
                step=epoch,
                mode=mode,
            )
        metrics = Metrics(ds=mouse_ds, results=mouse_result)

        trial_corr = metrics.single_trial_correlation(per_neuron=True)
        results["trial_correlation"][mouse_id] = trial_corr.mean()
        trial_corrs[mouse_id] = trial_corr

        if metrics.repeat_image and not metrics.hashed:
            image_corr = metrics.correlation_to_average(per_neuron=True)
            results["image_correlation"][mouse_id] = image_corr.mean()
            image_corrs[mouse_id] = image_corr
            feve = metrics.feve(per_neuron=True)
            results["feve"][mouse_id] = feve.mean()
            feves[mouse_id] = feve

    # write individual and average results to TensorBoard
    if summary is not None:
        summary.plot_correlation(
            "single_trial_correlation",
            data=metrics2df(trial_corrs),
            step=epoch,
            mode=mode,
        )
        if image_corrs:
            summary.plot_correlation(
                "correlation_to_average",
                data=metrics2df(image_corrs),
                step=epoch,
                mode=mode,
            )
        if feves:
            summary.plot_correlation(
                "FEVE",
                data=metrics2df(feves),
                step=epoch,
                ylabel="FEVE",
                mode=mode,
            )
        for metric, result in results.items():
            for mouse_id, value in result.items():
                summary.scalar(
                    tag=f"{metric}/mouse{mouse_id}",
                    value=value,
                    step=epoch,
                    mode=mode,
                )
            values = list(result.values())
            if values:
                summary.scalar(
                    tag=f"{metric}/average",
                    value=np.mean(values),
                    step=epoch,
                    mode=mode,
                )
    if print_result:
        _print = lambda d: [f"Mouse {k}: {v:.04f}\t\t" for k, v in d.items()]
        statement = "Single trial correlation\n"
        statement += "".join(_print(results["trial_correlation"]))
        if results["image_correlation"]:
            statement += "\nCorrelation to average\n"
            statement += "".join(_print(results["image_correlation"]))
            statement += "\nFEVE\n"
            statement += "".join(_print(results["feve"]))
        print(statement)
    if save_result is not None:
        yaml.save(os.path.join(save_result, "results.yaml"), data=results)
    return results


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


def metrics2df(results: t.Dict[int, torch.Tensor]):
    mouse_ids, values = [], []
    for mouse_id, v in results.items():
        mouse_ids.extend([mouse_id] * len(v))
        values.extend(v.tolist())
    return pd.DataFrame({"mouse": mouse_ids, "results": values})


def log_metrics(
    results: t.Dict[
        t.Union[str, int],
        t.Union[t.List[torch.Tensor], t.Dict[str, torch.Tensor]],
    ],
    epoch: int,
    mode: int,
    summary: tensorboard.Summary,
    mouse_id: int = None,
):
    """Compute the mean of the metrics in results and log to Summary

    Args:
        results: t.Dict[
                t.Union[str, int],
                t.Union[t.List[torch.Tensor], t.Dict[str, torch.Tensor]]
            ],
            a dictionary of tensors where keys are the name of the metrics
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
        metrics = list(results.keys())
        for metric in metrics:
            results[metric] = torch.stack(results[metric]).mean()
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


def num_steps(ds: t.Dict[int, DataLoader]):
    """Return the number of total steps to iterate all the DataLoaders"""
    return sum([len(ds[k]) for k in ds.keys()])
