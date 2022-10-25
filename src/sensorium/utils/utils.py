import os
import sys
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
from sensorium import losses, data
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
    mouse_id, device = ds.dataset.mouse_id, model.device
    model.train(False)
    with torch.no_grad():
        for data in ds:
            predictions = model(
                data["image"].to(device),
                mouse_id=mouse_id,
                pupil_center=data["pupil_center"].to(device),
            )
            results["predictions"].append(predictions.cpu())
            results["targets"].append(data["response"])
            results["images"].append(data["image"])
            results["image_ids"].append(data["image_id"])
            results["trial_ids"].append(data["trial_id"])
    results = {
        k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
        for k, v in results.items()
    }
    # convert images to their original range for plotting
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
) -> t.Dict[str, np.ndarray]:
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
        ds.items(), desc="Evaluation", disable=args.verbose < 2
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

    if args.verbose and print_result:
        _print = lambda d: [f"M{k}: {v:.04f}\t" for k, v in d.items()]
        statement = ""
        for metric in metrics:
            if results[metric]:
                statement += f"\n{metric}\n"
                statement += "".join(_print(results[metric]))
        if statement:
            print(statement)

    # compute overall average for each metric
    overall_result = {}
    for metric in metrics:
        values = list(results[metric].values())
        if values:
            average = np.mean(values)
            overall_result[metric] = average
            results[metric]["average"] = average
            if summary is not None:
                summary.scalar(
                    f"{metric}/average",
                    value=average,
                    step=epoch,
                    mode=mode,
                )

    if save_result is not None:
        yaml.save(os.path.join(save_result, "evaluation.yaml"), data=results)

    return overall_result


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
    try:
        arguments["git_hash"] = check_output(["git", "describe", "--always"])
        arguments["hostname"] = check_output(["hostname"])
    except subprocess.CalledProcessError as e:
        if args.verbose > 0:
            print(f"Unable to call subprocess: {e}")
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
    results: t.Dict[t.Union[int, str], t.Dict[str, t.Any]],
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
    """
    mouse_ids = list(results.keys())
    metrics = list(results[mouse_ids[0]].keys())
    for mouse_id in mouse_ids:
        for metric in metrics:
            value = results[mouse_id][metric]
            if not isinstance(value, float):
                results[mouse_id][metric] = np.mean(value)
            summary.scalar(
                f"{metric}/mouse{mouse_id}",
                value=results[mouse_id][metric],
                step=epoch,
                mode=mode,
            )
    overall_result = {}
    for metric in metrics:
        value = np.mean([results[mouse_id][metric] for mouse_id in mouse_ids])
        overall_result[metric[metric.find("/") + 1 :]] = value
        summary.scalar(metric, value=value, step=epoch, mode=mode)
    return overall_result


def num_steps(ds: t.Dict[int, DataLoader]):
    """Return the number of total steps to iterate all the DataLoaders"""
    return sum([len(ds[k]) for k in ds.keys()])


def load_pretrain_core(args, model: Model):
    filename = os.path.join(args.pretrain_core, "ckpt", "model_state.pt")
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


def get_batch_size(
    args,
    max_batch_size: int = None,
    num_iterations: int = 5,
):
    """
    Calculate the maximum batch size that can fill the GPU memory if CUDA device
    is set and args.batch_size is not set.
    """
    device = args.device.type

    if ("cuda" not in device) or ("cuda" in device and args.batch_size != 0):
        assert args.batch_size > 1
    else:
        device, mouse_id = args.device, 2
        train_ds, _, _ = data.get_training_ds(
            args,
            data_dir=args.dataset,
            mouse_ids=[mouse_id],
            batch_size=1,
            device=device,
        )

        output_shape = (train_ds[mouse_id].dataset.num_neurons,)
        model = Model(args, ds=train_ds)
        model.to(device)
        model.train(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = losses.get_criterion(args, ds=train_ds)

        ds_size = len(train_ds[mouse_id].dataset)

        batch_size = 1
        while True:
            if max_batch_size is not None and batch_size >= max_batch_size:
                batch_size = max_batch_size
                break
            if batch_size >= ds_size:
                batch_size //= 2
                break
            try:
                for _ in range(num_iterations):
                    targets = torch.rand(*(batch_size, *output_shape), device=device)
                    outputs = model(
                        torch.rand(*(batch_size, *args.input_shape), device=device),
                        mouse_id=mouse_id,
                        pupil_center=torch.rand(batch_size, 2, device=device),
                    )
                    loss = criterion(y_true=targets, y_pred=outputs, mouse_id=mouse_id)
                    loss += model.regularizer(mouse_id=mouse_id)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                batch_size *= 2
            except RuntimeError:
                if args.verbose > 1:
                    print(f"OOM at batch size: {batch_size}")
                batch_size //= 2
                break
        del train_ds, model, optimizer, criterion
        torch.cuda.empty_cache()
        if args.verbose > 1:
            print(f"set batch size: {batch_size}")
        args.batch_size = batch_size
