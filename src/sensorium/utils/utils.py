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

from sensorium.utils import yaml


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
        ds.items(), desc="Inference", disable=args.verbose == 0
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
            predictions = predictions.detach().cpu()
            predictions = dataloader.dataset.i_transform_response(predictions)
            result["predictions"].append(predictions)
            result["targets"].append(batch["response"])
            images = dataloader.dataset.i_transform_image(images.detach().cpu())
            result["images"].append(images)
            result["frame_ids"].append(batch["frame_id"])
            result["trial_ids"].append(batch["trial_id"])
        results[mouse_id] = {k: torch.cat(v, dim=0) for k, v in result.items()}
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


def get_available_device(no_acceleration: bool):
    device = torch.device("cpu")
    if not no_acceleration:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
    return device


def metrics2df(results: t.Dict[str, torch.Tensor]):
    mouse_ids, values = [], []
    for mouse_id, v in results.items():
        mouse_ids.extend([mouse_id] * len(v))
        values.extend(v.tolist())
    return pd.DataFrame({"mouse": mouse_ids, "results": values})


def save_checkpoint(args, model: nn.Module, optimizer: torch.optim, epoch: int):
    if not os.path.isdir(ckpt_dir := os.path.join(args.output_dir, "ckpt")):
        os.makedirs(ckpt_dir)
    filename = os.path.join(ckpt_dir, f"epoch-{epoch:03d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f=filename,
    )
    if args.verbose:
        print(f"Checkpoint saved to {filename}.")


def load_checkpoint(args, model: nn.Module, optimizer: torch.optim):
    epoch = 0
    if not os.path.isdir(ckpt_dir := os.path.join(args.output_dir, "ckpt")):
        os.makedirs(ckpt_dir)
    checkpoints = sorted(glob(os.path.join(ckpt_dir, "epoch-*.pt")))
    if checkpoints:
        # load last checkpoint
        checkpoint = torch.load(checkpoints[-1])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        if args.verbose:
            print(f"Loaded checkpoint from {checkpoint[-1]}.")
    return epoch
