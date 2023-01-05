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


@torch.no_grad()
def inference(
    ds: DataLoader,
    model: torch.nn.Module,
    micro_batch_size: int,
    device: torch.device = "cpu",
) -> t.Dict[str, torch.Tensor]:
    """Inference data in DataLoader ds
    Returns:
        results: t.Dict[int, t.Dict[str, torch.Tensor]]
            - mouse_id
                - predictions: torch.Tensor, predictions given images
                - targets: torch.Tensor, the ground-truth responses
                - trial_ids: torch.Tensor, trial ID of the responses
                - image_ids: torch.Tensor, image ID of the responses
    """
    results = {
        "predictions": [],
        "targets": [],
        "trial_ids": [],
        "image_ids": [],
    }
    mouse_id = ds.dataset.mouse_id
    model.to(device)
    model.train(False)
    for batch in ds:
        for micro_batch in data.micro_batching(batch, batch_size=micro_batch_size):
            predictions, _, _ = model(
                inputs=micro_batch["image"].to(device),
                mouse_id=mouse_id,
                behaviors=micro_batch["behavior"].to(device),
                pupil_centers=micro_batch["pupil_center"].to(device),
            )
            results["predictions"].append(predictions.cpu())
            results["targets"].append(micro_batch["response"])
            results["image_ids"].append(micro_batch["image_id"])
            results["trial_ids"].append(micro_batch["trial_id"])
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

    batch_size = args.batch_size
    if hasattr(args, "micro_batch_size"):
        batch_size = args.micro_batch_size

    for mouse_id, mouse_ds in tqdm(
        ds.items(), desc="Evaluation", disable=args.verbose < 2
    ):
        if mouse_id in (0, 1) and mouse_ds.dataset.tier == "test":
            continue
        outputs[mouse_id] = inference(
            ds=mouse_ds,
            model=model,
            micro_batch_size=batch_size,
            device=args.device,
        )

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
        # create box plot for each metric
        for metric, values in results.items():
            if values:
                summary.box_plot(
                    metric,
                    data=metrics2df(results[metric]),
                    step=epoch,
                    mode=mode,
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


@torch.no_grad()
def plot_samples(
    args,
    model: nn.Module,
    ds: t.Dict[int, DataLoader],
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
    num_samples: int = 5,
):
    device = args.device
    model.to(device)
    model.train(False)
    for mouse_id, mouse_ds in ds.items():
        results = {
            "images": [],
            "crop_images": [],
            "image_grids": [],
            "targets": [],
            "predictions": [],
            "pupil_center": [],
            "behaviors": [],
            "image_ids": [],
        }
        i_transform_image = mouse_ds.dataset.i_transform_image
        for batch in mouse_ds:
            for micro_batch in data.micro_batching(
                batch, batch_size=args.micro_batch_size
            ):
                images = micro_batch["image"]
                predictions, crop_images, image_grids = model(
                    inputs=images.to(device),
                    mouse_id=mouse_id,
                    pupil_centers=micro_batch["pupil_center"].to(device),
                    behaviors=micro_batch["behavior"].to(device),
                )
                images = i_transform_image(images)
                crop_images = i_transform_image(crop_images.cpu())
                image_grids = image_grids.cpu()
                predictions = predictions.cpu()
                for i in range(len(predictions)):
                    results["images"].append(images[i])
                    results["crop_images"].append(crop_images[i])
                    results["image_grids"].append(image_grids[i])
                    results["targets"].append(micro_batch["response"][i])
                    results["predictions"].append(predictions[i])
                    results["pupil_center"].append(micro_batch["pupil_center"][i])
                    results["behaviors"].append(micro_batch["behavior"][i])
                    results["image_ids"].append(micro_batch["image_id"][i])
                    if len(results["images"]) == num_samples:
                        break
                if len(results["images"]) == num_samples:
                    break
            if len(results["images"]) == num_samples:
                break
        results = {k: torch.stack(v, dim=0).numpy() for k, v in results.items()}
        summary.plot_image_response(
            f"image_response/mouse{mouse_id}", results=results, step=epoch, mode=mode
        )


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
            # allow TensorFloat32 computation
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
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
    summary: tensorboard.Summary = None,
    mode: int = 0,
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
            if isinstance(value, list):
                if torch.is_tensor(value[0]):
                    results[mouse_id][metric] = torch.mean(torch.stack(value)).item()
                else:
                    results[mouse_id][metric] = np.mean(value)
            if summary is not None:
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
        if summary is not None:
            summary.scalar(metric, value=value, step=epoch, mode=mode)
    return overall_result


def num_steps(ds: t.Dict[int, DataLoader]):
    """Return the number of total steps to iterate all the DataLoaders"""
    return sum([len(ds[k]) for k in ds.keys()])


def load_pretrain_core(args, model: Model, device: torch.device = "cpu"):
    filename = os.path.join(args.pretrain_core, "ckpt", "model_state.pt")
    assert os.path.exists(filename), f"Cannot find pretrain core {filename}."
    model_dict = model.state_dict()
    core_ckpt = torch.load(filename, map_location=device)
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


def compute_micro_batch_size(
    args, batch_iterations: int = 5, micro_iterations: int = 5
):
    """
    Calculate the maximum micro batch size that can fill the GPU memory if
    CUDA device is set.
    """
    if hasattr(args, "micro_batch_size") and args.micro_batch_size:
        assert args.micro_batch_size <= args.batch_size
        return

    device = args.device
    if "cuda" not in device.type:
        args.micro_batch_size = args.batch_size
        return

    mouse_ids = args.mouse_ids
    # create dummy dataloaders, model, optimizer and criterion
    train_ds, _, _ = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=mouse_ids,
        batch_size=1,
        device=device,
    )
    model = Model(args, ds=train_ds)
    model.to(device)
    model.train(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = losses.get_criterion(args, ds=train_ds)

    image_shape = args.input_shape
    random_input = lambda size: torch.rand(*size, device=device)

    batch_size, micro_batch_size = args.batch_size, 1
    while True:
        if micro_batch_size >= batch_size:
            micro_batch_size = batch_size
            break
        try:
            # dummy training loop to mimic training and gradient accumulation
            for _ in range(batch_iterations):
                for mouse_id in mouse_ids:
                    batch_loss = 0.0
                    for _ in range(micro_iterations):
                        outputs, _, _ = model(
                            inputs=random_input((micro_batch_size, *image_shape)),
                            mouse_id=mouse_id,
                            behaviors=random_input((micro_batch_size, 3)),
                            pupil_centers=random_input((micro_batch_size, 2)),
                        )
                        loss = criterion(
                            y_true=random_input((micro_batch_size, *outputs.shape)),
                            y_pred=outputs,
                            mouse_id=mouse_id,
                            batch_size=batch_size,
                        )
                        batch_loss += loss
                    reg_loss = model.regularizer(mouse_id=mouse_id)
                    total_loss = batch_loss + reg_loss
                    total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            micro_batch_size += 7 if micro_batch_size == 1 else 8
        except RuntimeError:
            if args.verbose:
                print(f"OOM at micro batch size {micro_batch_size}")
            micro_batch_size -= 7 if micro_batch_size == 8 else 8
            break
    del train_ds, model, optimizer, criterion
    torch.cuda.empty_cache()

    assert micro_batch_size > 0
    if args.verbose:
        print(f"set micro batch size to {micro_batch_size}")
    args.micro_batch_size = micro_batch_size


class AutoGradClip:
    """
    Automatic gradient clipping
    reference:
    - https://arxiv.org/abs/2007.14469
    - https://github.com/pseeth/autoclip
    """

    def __init__(self, percentile: float, max_history: int = 10000):
        assert 0 <= percentile <= 100
        self.idx = 0
        self.percentile = percentile
        self.max_history = max_history
        self.history = np.zeros(shape=(max_history,), dtype=np.float32)

    @staticmethod
    def compute_grad_norm(model: nn.Module):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1.0 / 2)

    def __call__(self, model: nn.Module):
        grad_norm = self.compute_grad_norm(model)
        self.history[self.idx % self.max_history] = grad_norm
        self.idx += 1
        max_norm = np.percentile(self.history[: self.idx], q=self.percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
