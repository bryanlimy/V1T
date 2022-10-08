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

from sensorium.models import Model
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
    mouse_id = ds.dataset.mouse_id
    model.train(False)
    with torch.no_grad():
        for data in ds:
            predictions = model(data["image"].to(model.device), mouse_id=mouse_id)
            results["predictions"].append(predictions.cpu())
            results["targets"].append(data["response"])
            results["images"].append(data["image"])
            results["image_ids"].append(data["image_id"])
            results["trial_ids"].append(data["trial_id"])
    results = {
        k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
        for k, v in results.items()
    }
    # convert responses and images to desired range for evaluation and plotting
    results["predictions"] = ds.dataset.transform4evaluation(results["predictions"])
    results["targets"] = ds.dataset.transform4evaluation(results["targets"])
    results["images"] = ds.dataset.i_transform_image(results["images"])
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
    metrics = ["single_trial_correlation", "correlation_to_average", "feve"]
    outputs, results = {}, {k: {} for k in metrics}

    for mouse_id, mouse_ds in tqdm(
        ds.items(), desc="Evaluation", disable=args.verbose == 0
    ):
        if mouse_id in (0, 1) and mouse_ds.dataset.tier == "test":
            continue
        outputs[mouse_id] = inference(ds=mouse_ds, model=model)

        mouse_metric = Metrics(ds=mouse_ds, results=outputs[mouse_id])

        results["single_trial_correlation"][
            mouse_id
        ] = mouse_metric.single_trial_correlation(per_neuron=True)
        if mouse_metric.repeat_image and not mouse_metric.hashed:
            results["correlation_to_average"][
                mouse_id
            ] = mouse_metric.correlation_to_average(per_neuron=True)
            results["feve"][mouse_id] = mouse_metric.feve(per_neuron=True)

    if summary is not None:
        # create image-response pair plots
        for mouse_id in outputs.keys():
            summary.plot_image_response(
                tag=f"image_response/mouse{mouse_id}",
                results=outputs[mouse_id],
                step=epoch,
                mode=mode,
            )
        # create box plot for each metric
        for metric, values in results.items():
            if values:
                summary.box_plot(
                    metric, data=metrics2df(results[metric]), step=epoch, mode=mode
                )

    # compute the average value for each mouse
    for metric in metrics:
        for mouse_id in results[metric].keys():
            results[metric][mouse_id] = np.mean(results[metric][mouse_id])
            if summary is not None:
                summary.scalar(
                    f"{metric}/mouse{mouse_id}",
                    value=results[metric][mouse_id],
                    step=epoch,
                    mode=mode,
                )

    if print_result:
        _print = lambda d: [f"M{k}: {v:.04f}\t" for k, v in d.items()]
        statement = ""
        for metric in metrics:
            if results[metric]:
                statement += f"\n{metric}\n"
                statement += "".join(_print(results[metric]))
        if statement:
            print(statement)

    # compute overall average for each metric
    for metric in metrics:
        values = list(results[metric].values())
        if values:
            results[metric]["average"] = np.mean(values)
            if summary is not None:
                summary.scalar(
                    f"{metric}/average",
                    value=results[metric]["average"],
                    step=epoch,
                    mode=mode,
                )

    if save_result is not None:
        yaml.save(os.path.join(save_result, "evaluation.yaml"), data=results)
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
    results: t.Dict[t.Union[int, str], t.Dict[str, t.List[float]]],
    epoch: int,
    mode: int,
    summary: tensorboard.Summary,
):
    """Compute the mean of the metrics in results and log to Summary

    Args:
        results: t.Dict[t.Union[int, str], t.Dict[str, t.List[float]]],
            a dictionary of tensors where keys are the name of the metrics
            that represent results from of a mouse.
        epoch: int, the current epoch number.
        mode: int, Summary logging mode.
        summary: tensorboard.Summary, Summary class
        mouse_id: int, the mouse_id of the result dictionary, None if the
            dictionary represents results from multiple mice.
    """
    keys = list(results.keys())
    metrics = list(results[keys[0]].keys())
    for mouse_id in keys:
        for metric in metrics:
            results[mouse_id][metric] = np.mean(results[mouse_id][metric])
            summary.scalar(
                f"{metric}/mouse{mouse_id}",
                value=results[mouse_id][metric],
                step=epoch,
                mode=mode,
            )
    for metric in metrics:
        results[metric] = np.mean([results[mouse_id][metric] for mouse_id in keys])
        summary.scalar(metric, value=results[metric], step=epoch, mode=mode)


def num_steps(ds: t.Dict[int, DataLoader]):
    """Return the number of total steps to iterate all the DataLoaders"""
    return sum([len(ds[k]) for k in ds.keys()])


def load_pretrain_core(args, model: Model):
    filename = os.path.join(args.pretrain_core, "ckpt", "best_model.pt")
    assert os.path.exists(filename), f"Cannot find pretrain core {filename}."
    model_dict = model.state_dict()
    core_ckpt = torch.load(filename, map_location=model.device)
    # add 'core.' to parameters in pretrained core
    core_dict = {f"core.{k}": v for k, v in core_ckpt["model_state_dict"].items()}
    # check pretrained core has the same parameters in core module
    for key in model_dict.keys():
        if key.startswith("core."):
            assert key in core_dict
    model_dict.update(core_dict)
    model.load_state_dict(model_dict)
    if args.verbose:
        print(f"\nLoaded pretrained core from {args.pretrain_core}.\n")


def load_checkpoint(args, model: Model):
    filename = os.path.join(args.output_dir, "ckpt", "best_model.pt")
    assert os.path.exists(filename), f"Cannot find model checkpoint {filename}."
    ckpt = torch.load(filename, map_location=model.device)
    epoch = ckpt["epoch"]
    model.load_state_dict(ckpt["model_state_dict"])
    if args.verbose:
        print(f"\nLoaded checkpoint (epoch {epoch}) from {filename}.\n")
