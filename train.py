import os
import torch
import wandb
import argparse
import numpy as np
import typing as t
from torch import nn
from tqdm import tqdm
from time import time
from shutil import rmtree
from torch.utils.data import DataLoader

from sensorium import losses, data
from sensorium.models import get_model
from sensorium.utils.logger import Logger
from sensorium.utils import utils, tensorboard
from sensorium.utils.scheduler import Scheduler


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Metrics to compute as part of training and validation step"""
    msse = losses.msse(y_true=y_true, y_pred=y_pred)
    poisson_loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred)
    correlation = losses.correlation(y1=y_pred, y2=y_true, dim=None)
    return {
        "metrics/msse": msse.item(),
        "metrics/poisson_loss": poisson_loss.item(),
        "metrics/single_trial_correlation": correlation.item(),
    }


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


def train_step(
    mouse_id: int,
    batch: t.Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim,
    criterion: losses.Loss,
    update: bool,
    grad_clip: AutoGradClip = None,
    device: torch.device = "cpu",
) -> (t.Dict[str, torch.Tensor], t.Dict[str, float]):
    model.to(device)
    responses = batch["response"].to(device)
    outputs, _, _ = model(
        inputs=batch["image"].to(device),
        mouse_id=mouse_id,
        behaviors=batch["behavior"].to(device),
        pupil_centers=batch["pupil_center"].to(device),
    )
    loss = criterion(y_true=responses, y_pred=outputs, mouse_id=mouse_id)
    reg_loss = model.regularizer(mouse_id=mouse_id)
    total_loss = loss + reg_loss
    total_loss.backward()  # calculate and accumulate gradients
    module_grad_norms = None
    if update:
        # if grad_clip is not None:
        #     grad_clip(model)
        module_grad_norms = {
            "grad_norm/image_cropper": grad_clip.compute_grad_norm(model.image_cropper),
            "grad_norm/core": grad_clip.compute_grad_norm(model.core),
            "grad_norm/core_shifter": grad_clip.compute_grad_norm(model.core_shifter),
            f"grad_norm/mouse{mouse_id}_readout": grad_clip.compute_grad_norm(
                model.readouts[str(mouse_id)]
            ),
        }
        # total_norm = grad_clip.compute_grad_norm(model)
        # print(f"total_norm: {total_norm:.02f}")
        optimizer.step()
        optimizer.zero_grad()
    result = {
        "loss/loss": loss.item(),
        "loss/reg_loss": reg_loss.item(),
        "loss/total_loss": total_loss.item(),
        **compute_metrics(y_true=responses.detach(), y_pred=outputs.detach()),
    }
    return result, module_grad_norms


def train(
    args,
    ds: t.Dict[int, DataLoader],
    model: nn.Module,
    optimizer: torch.optim,
    criterion: losses.Loss,
    grad_clip: AutoGradClip,
    epoch: int,
    summary: tensorboard.Summary,
) -> t.Dict[t.Union[str, int], t.Union[torch.Tensor, t.Dict[str, torch.Tensor]]]:
    mouse_ids = list(ds.keys())
    results = {mouse_id: {} for mouse_id in mouse_ids}
    ds = data.CycleDataloaders(ds)
    # call optimizer.step() after iterate one batch from each mouse
    update_frequency = len(mouse_ids)
    model.train(True)
    model.requires_grad_(True)
    optimizer.zero_grad()
    for i, (mouse_id, mouse_batch) in tqdm(
        enumerate(ds), desc="Train", total=len(ds), disable=args.verbose < 2
    ):
        result, grad_norms = train_step(
            mouse_id=mouse_id,
            batch=mouse_batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            update=(i + 1) % update_frequency == 0,
            grad_clip=grad_clip,
            device=args.device,
        )
        utils.update_dict(results[mouse_id], result)
        if grad_norms is not None:
            for m, grad_norm in grad_norms.items():
                summary.scalar(m, value=grad_norm, step=epoch * (i + 1))
    return utils.log_metrics(results=results, epoch=epoch, mode=0, summary=summary)


def validation_step(
    mouse_id: int,
    batch: t.Dict[str, torch.Tensor],
    model: nn.Module,
    criterion: losses.Loss,
    device: torch.device = "cpu",
) -> t.Dict[str, torch.Tensor]:
    result = {}
    model.to(device)
    responses = batch["response"].to(device)
    outputs, _, _ = model(
        inputs=batch["image"].to(device),
        mouse_id=mouse_id,
        behaviors=batch["behavior"].to(device),
        pupil_centers=batch["pupil_center"].to(device),
    )
    loss = criterion(y_true=responses, y_pred=outputs, mouse_id=mouse_id)
    result["loss/loss"] = loss.item()
    result.update(compute_metrics(y_true=responses, y_pred=outputs))
    return result


@torch.no_grad()
def validate(
    args,
    ds: t.Dict[int, DataLoader],
    model: nn.Module,
    criterion: losses.Loss,
    epoch: int,
    summary: tensorboard.Summary,
) -> t.Dict[t.Union[str, int], t.Union[torch.Tensor, t.Dict[str, torch.Tensor]]]:
    model.train(False)
    results = {}
    with tqdm(desc="Val", total=utils.num_steps(ds), disable=args.verbose < 2) as pbar:
        for mouse_id, mouse_ds in ds.items():
            mouse_result = {}
            for batch in mouse_ds:
                result = validation_step(
                    mouse_id=mouse_id,
                    batch=batch,
                    model=model,
                    criterion=criterion,
                    device=args.device,
                )
                utils.update_dict(mouse_result, result)
                pbar.update(1)
            results[mouse_id] = mouse_result
    return utils.log_metrics(results=results, epoch=epoch, mode=1, summary=summary)


def main(args, wandb_sweep: bool = False):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.use_wandb:
        os.environ["WANDB_SILENT"] = "true"
        if not wandb_sweep:
            try:
                wandb.init(
                    config=args,
                    dir=os.path.join(args.output_dir, "wandb"),
                    project="sensorium",
                    entity="bryanlimy",
                    group=args.wandb_group,
                    name=os.path.basename(args.output_dir),
                )
            except AssertionError as e:
                print(f"wandb.init error: {e}")
                args.use_wandb = False

    Logger(args)
    utils.set_random_seed(args.seed)
    utils.get_device(args)

    if not args.mouse_ids:
        args.mouse_ids = list(range(1 if args.behavior_mode else 0, 7))
    if args.batch_size == 0 and "cuda" in args.device.type:
        utils.auto_batch_size(args)
        if args.use_wandb:
            wandb.config.update({"batch_size": args.batch_size}, allow_val_change=True)

    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    summary = tensorboard.Summary(args)

    model = get_model(args, ds=train_ds, summary=summary)

    if args.pretrain_core:
        utils.load_pretrain_core(args, model=model)

    optimizer = torch.optim.AdamW(
        params=model.get_parameters(core_lr=args.core_lr_scale * args.lr),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=0,
    )
    scheduler = Scheduler(args, model=model, optimizer=optimizer, mode="max")
    criterion = losses.get_criterion(args, ds=train_ds)
    grad_clip = AutoGradClip(percentile=10)

    utils.save_args(args)

    epoch = scheduler.restore(load_optimizer=True, load_scheduler=True)

    # utils.plot_samples(
    #     model,
    #     ds=train_ds,
    #     summary=summary,
    #     epoch=epoch,
    #     device=args.device,
    # )

    while (epoch := epoch + 1) < args.epochs + 1:
        if args.verbose:
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_result = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            grad_clip=grad_clip,
            epoch=epoch,
            summary=summary,
        )
        val_result = validate(
            args,
            ds=val_ds,
            model=model,
            criterion=criterion,
            epoch=epoch,
            summary=summary,
        )
        elapse = time() - start

        summary.scalar("model/elapse", value=elapse, step=epoch, mode=0)
        for param_group in optimizer.param_groups:
            summary.scalar(
                f'model/lr/{param_group["name"] if "name" in param_group else "model"}',
                value=param_group["lr"],
                step=epoch,
                mode=0,
            )
        if epoch % 10 == 0:
            utils.plot_samples(
                model,
                ds=val_ds,
                summary=summary,
                epoch=epoch,
                device=args.device,
            )
        if args.verbose:
            print(
                f'Train\t\t\tloss: {train_result["loss"]:.04f}\t\t'
                f'correlation: {train_result["single_trial_correlation"]:.04f}\n'
                f'Validation\t\tloss: {val_result["loss"]:.04f}\t\t'
                f'correlation: {val_result["single_trial_correlation"]:.04f}\n'
                f"Elapse: {elapse:.02f}s"
            )
        early_stop = scheduler.step(val_result["single_trial_correlation"], epoch=epoch)
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": train_result["loss"],
                    "train_corr": train_result["single_trial_correlation"],
                    "val_loss": val_result["loss"],
                    "val_corr": val_result["single_trial_correlation"],
                    "best_corr": scheduler.best_value,
                },
                step=epoch,
            )
        if np.isnan(train_result["loss"]) or np.isnan(val_result["loss"]):
            if args.use_wandb:
                wandb.finish(exit_code=1)  # mark run as failed
            exit("\nNaN loss detected, determinate training.")
        if early_stop:
            break

    scheduler.restore()
    eval_result = utils.evaluate(
        args,
        ds=test_ds,
        model=model,
        epoch=epoch,
        summary=summary,
        mode=2,
        print_result=True,
        save_result=args.output_dir,
    )
    if args.use_wandb:
        wandb.log({"test_corr": eval_result["single_trial_correlation"]}, step=epoch)
    utils.plot_samples(
        model,
        ds=test_ds,
        summary=summary,
        epoch=epoch,
        mode=2,
        device=args.device,
    )
    if args.verbose:
        print(f"\nResults saved to {args.output_dir}.")
    summary.close()
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sensorium",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mouse_ids",
        nargs="+",
        type=int,
        default=None,
        help="Mouse to use for training.",
    )
    parser.add_argument(
        "--behavior_mode",
        required=True,
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="behavior mode:"
        "0: do not include behavior"
        "1: concat behavior with natural image"
        "2: add latent behavior variables to each ViT block"
        "3: add latent behavior + pupil centers to each ViT block"
        "4: separate BehaviorMLP for each animal",
    )
    parser.add_argument(
        "--center_crop",
        default=1.0,
        type=float,
        help="crop the center of the image to (scale * height, scale, width)",
    )
    parser.add_argument(
        "--resize_image",
        default=1,
        type=int,
        choices=[0, 1],
        help="resize image mode:"
        "0: no resizing, return full image (1, 144, 256)"
        "1: resize image to (1, 36, 64)",
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="number of works for DataLoader.",
    )

    # training settings
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="maximum epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="If batch_size == 0 and CUDA is available, then dynamically test "
        "batch size. Otherwise use the provided value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="",
        help="Device to use for computation. "
        "Use the best available device if --device is not specified.",
    )
    parser.add_argument("--seed", type=int, default=1234)

    # optimizer settings
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.9999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument(
        "--criterion",
        default="poisson",
        type=str,
        help="criterion (loss function) to use.",
    )

    parser.add_argument(
        "--ds_scale",
        action="store_true",
        help="scale loss by the size of the dataset",
    )

    # pre-trained Core
    parser.add_argument(
        "--pretrain_core",
        type=str,
        default="",
        help="path to directory where pre-trained core model is stored.",
    )
    parser.add_argument(
        "--core_lr_scale",
        type=float,
        default=1,
        help="scale learning rate for core as it might already be trained.",
    )

    # plot settings
    parser.add_argument(
        "--save_plots", action="store_true", help="save plots to --output_dir"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="matplotlib figure DPI",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="svg",
        choices=["pdf", "svg", "png"],
        help="file format when --save_plots",
    )

    # wandb settings
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_group", type=str, default="")

    # misc
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])

    # model settings
    parser.add_argument(
        "--core",
        type=str,
        required=True,
        help="The core module to use.",
    )
    parser.add_argument(
        "--readout",
        type=str,
        required=True,
        help="The readout module to use.",
    )
    parser.add_argument(
        "--shift_mode",
        type=int,
        default=2,
        choices=[0, 1, 2, 3, 4],
        help="shift mode: "
        "0 - disable shifter, "
        "1 - shift input to core module, "
        "2 - shift input to readout module"
        "3 - shift input to both core and readout module"
        "4 - shift_mode=3 and provide both behavior and pupil center to cropper",
    )

    temp_args = parser.parse_known_args()[0]

    # hyper-parameters for core module
    if temp_args.core == "conv":
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--num_filters", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--core_reg_scale", type=float, default=0)
        parser.add_argument("--lr", default=0.001, type=float)
    elif temp_args.core == "stacked2d":
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--core_reg_input", type=float, default=6.3831)
        parser.add_argument("--core_reg_hidden", type=float, default=0.0)
        parser.add_argument("--lr", default=0.009, type=float)
    elif temp_args.core == "vit":
        parser.add_argument("--patch_size", type=int, default=8)
        parser.add_argument(
            "--patch_mode",
            type=int,
            default=0,
            choices=[0, 1, 2],
            help="patch embedding mode:"
            "0 - nn.Unfold to extract patches"
            "1 - nn.Conv2D to extract patches"
            "2 - Shifted Patch Tokenization https://arxiv.org/abs/2112.13492v1",
        )
        parser.add_argument(
            "--patch_stride",
            type=int,
            default=1,
            help="stride size to extract patches",
        )
        parser.add_argument("--num_blocks", type=int, default=4)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--emb_dim", type=int, default=155)
        parser.add_argument("--mlp_dim", type=int, default=488)
        parser.add_argument(
            "--p_dropout", type=float, default=0.0229, help="patch embeddings dropout"
        )
        parser.add_argument(
            "--t_dropout", type=float, default=0.2544, help="ViT block dropout"
        )
        parser.add_argument(
            "--drop_path", type=float, default=0.0, help="stochastic depth dropout rate"
        )
        parser.add_argument(
            "--use_lsa", action="store_true", help="Use Locality Self Attention"
        )
        parser.add_argument("--core_reg_scale", type=float, default=0.5379)
        parser.add_argument("--lr", default=0.001647, type=float)
    elif temp_args.core == "cct":
        parser.add_argument("--patch_size", type=int, default=8)
        parser.add_argument(
            "--patch_stride",
            type=int,
            default=1,
            help="stride size to extract patches",
        )
        parser.add_argument("--num_blocks", type=int, default=4)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--emb_dim", type=int, default=160)
        parser.add_argument("--mlp_ratio", type=float, default=3)
        parser.add_argument(
            "--pos_emb", type=str, default="sine", choices=["sine", "learn", "none"]
        )
        parser.add_argument(
            "--p_dropout", type=float, default=0.0229, help="patch embeddings dropout"
        )
        parser.add_argument(
            "--t_dropout", type=float, default=0.2544, help="ViT block dropout"
        )
        parser.add_argument(
            "--drop_path", type=float, default=0.0, help="stochastic depth dropout rate"
        )
        parser.add_argument("--core_reg_scale", type=float, default=0)
        parser.add_argument("--lr", default=0.001, type=float)
    elif temp_args.core == "stn":
        parser.add_argument("--num_layers", type=int, default=7)
        parser.add_argument("--num_filters", type=int, default=63)
        parser.add_argument("--dropout", type=float, default=0.1135)
        parser.add_argument("--core_reg_scale", type=float, default=0.0450)
        parser.add_argument("--lr", default=0.001, type=float)
    else:
        parser.add_argument("--core_reg_scale", type=float, default=0)

    # hyper-parameters for readout modules
    if temp_args.readout == "gaussian2d":
        parser.add_argument("--disable_grid_predictor", action="store_true")
        parser.add_argument("--grid_predictor_dim", type=int, default=2, choices=[2, 3])
        parser.add_argument(
            "--bias_mode",
            type=int,
            default=0,
            choices=[0, 1, 2],
            help="Gaussian2d readout bias mode:"
            "0: initialize bias with zeros"
            "1: initialize bias with the mean responses"
            "2: initialize bias with the mean responses divide by standard deviation",
        )
        parser.add_argument("--readout_reg_scale", type=float, default=0.0076)
    else:
        parser.add_argument("--readout_reg_scale", type=float, default=0.0)

    # hyper-parameters for core shifter module
    if temp_args.shift_mode in (1, 2, 3, 4):
        parser.add_argument("--shifter_reg_scale", type=float, default=0.0)
    # hyper-parameters for image cropper module
    if temp_args.shift_mode in (2, 3, 4):
        parser.add_argument("--cropper_reg_scale", type=float, default=0.0)

    del temp_args

    main(parser.parse_args())
