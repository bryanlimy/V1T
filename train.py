import os
import torch
import wandb
import argparse
import numpy as np
import typing as t
from tqdm import tqdm
from time import time
from shutil import rmtree
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from v1t import losses, data
from v1t.utils.logger import Logger
from v1t.models import get_model, Model
from v1t.utils import utils, tensorboard
from v1t.utils.scheduler import Scheduler


@torch.no_grad()
def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Metrics to compute as part of training and validation step"""
    msse = losses.msse(y_true=y_true, y_pred=y_pred)
    poisson_loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred)
    correlation = losses.correlation(y1=y_pred, y2=y_true, dim=None)
    return {
        "metrics/msse": msse,
        "metrics/poisson_loss": poisson_loss,
        "metrics/single_trial_correlation": correlation,
    }


def train_step(
    mouse_id: str,
    batch: t.Dict[str, torch.Tensor],
    model: Model,
    optimizer: torch.optim,
    criterion: losses.Loss,
    scaler: GradScaler,
    update: bool,
    micro_batch_size: int,
    device: torch.device = "cpu",
) -> t.Dict[str, torch.Tensor]:
    model.to(device)
    batch_size = batch["image"].size(0)
    result = {"loss/loss": [], "loss/reg_loss": [], "loss/total_loss": []}
    targets, predictions = [], []
    for micro_batch in data.micro_batching(batch, micro_batch_size):
        with autocast(enabled=scaler.is_enabled(), dtype=torch.float16):
            y_true = micro_batch["response"].to(device)
            y_pred, _, _ = model(
                inputs=micro_batch["image"].to(device),
                mouse_id=mouse_id,
                behaviors=micro_batch["behavior"].to(device),
                pupil_centers=micro_batch["pupil_center"].to(device),
            )
            loss = criterion(
                y_true=y_true,
                y_pred=y_pred,
                mouse_id=mouse_id,
                batch_size=batch_size,
            )
            reg_loss = (y_true.size(0) / batch_size) * model.regularizer(mouse_id)
            total_loss = loss + reg_loss
        scaler.scale(total_loss).backward()
        result["loss/loss"].append(loss.detach().cpu())
        result["loss/reg_loss"].append(reg_loss.detach().cpu())
        result["loss/total_loss"].append(total_loss.detach().cpu())
        targets.append(y_true.detach().cpu())
        predictions.append(y_pred.detach().cpu())
        del micro_batch, y_true, y_pred
    if update:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    result = {k: torch.sum(torch.stack(v)) for k, v in result.items()}
    result.update(
        compute_metrics(
            y_true=torch.vstack(targets),
            y_pred=torch.vstack(predictions),
        )
    )
    return result


def train(
    args,
    ds: t.Dict[str, DataLoader],
    model: Model,
    optimizer: torch.optim,
    criterion: losses.Loss,
    scaler: GradScaler,
    epoch: int,
    summary: tensorboard.Summary,
) -> t.Dict[t.Union[str, int], t.Union[torch.Tensor, t.Dict[str, torch.Tensor]]]:
    mouse_ids = list(ds.keys())
    results = {mouse_id: {} for mouse_id in mouse_ids}
    ds = data.CycleDataloaders(ds)
    # accumulate gradients over all mouse for one batch
    update_frequency = len(mouse_ids)
    model.train(True)
    optimizer.zero_grad()
    for i, (mouse_id, mouse_batch) in tqdm(
        enumerate(ds), desc="Train", total=len(ds), disable=args.verbose < 2
    ):
        result = train_step(
            mouse_id=mouse_id,
            batch=mouse_batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            update=(i + 1) % update_frequency == 0,
            micro_batch_size=args.micro_batch_size,
            device=args.device,
        )
        utils.update_dict(results[mouse_id], result)
    return utils.log_metrics(results, epoch=epoch, summary=summary, mode=0)


@torch.no_grad()
def validation_step(
    mouse_id: str,
    batch: t.Dict[str, torch.Tensor],
    model: Model,
    criterion: losses.Loss,
    scaler: GradScaler,
    micro_batch_size: int,
    device: torch.device = "cpu",
) -> t.Dict[str, torch.Tensor]:
    model.to(device)
    batch_size = batch["image"].size(0)
    result = {"loss/loss": [], "loss/reg_loss": [], "loss/total_loss": []}
    targets, predictions = [], []
    for micro_batch in data.micro_batching(batch, micro_batch_size):
        with autocast(enabled=scaler.is_enabled(), dtype=torch.float16):
            y_true = micro_batch["response"].to(device)
            y_pred, _, _ = model(
                inputs=micro_batch["image"].to(device),
                mouse_id=mouse_id,
                behaviors=micro_batch["behavior"].to(device),
                pupil_centers=micro_batch["pupil_center"].to(device),
            )
            loss = criterion(
                y_true=y_true,
                y_pred=y_pred,
                mouse_id=mouse_id,
                batch_size=batch_size,
            )
            reg_loss = (y_true.size(0) / batch_size) * model.regularizer(mouse_id)
            total_loss = loss + reg_loss
        result["loss/loss"].append(loss.detach())
        result["loss/reg_loss"].append(reg_loss.detach())
        result["loss/total_loss"].append(total_loss.detach())
        targets.append(y_true.detach())
        predictions.append(y_pred.detach())
    result = {k: torch.sum(torch.stack(v)) for k, v in result.items()}
    result.update(
        compute_metrics(
            y_true=torch.vstack(targets),
            y_pred=torch.vstack(predictions),
        )
    )
    return result


def validate(
    args,
    ds: t.Dict[str, DataLoader],
    model: Model,
    criterion: losses.Loss,
    scaler: GradScaler,
    epoch: int,
    summary: tensorboard.Summary = None,
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
                    scaler=scaler,
                    micro_batch_size=args.micro_batch_size,
                    device=args.device,
                )
                utils.update_dict(mouse_result, result)
                pbar.update(1)
            results[mouse_id] = mouse_result
    return utils.log_metrics(results, epoch=epoch, summary=summary, mode=1)


def main(args, wandb_sweep: bool = False):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    Logger(args)
    utils.get_device(args)
    utils.set_random_seed(args.seed, deterministic=args.deterministic)

    data.get_mouse_ids(args)

    if args.grad_checkpointing is None and args.core in ("vit", "cct"):
        args.grad_checkpointing = "cuda" in args.device.type
    utils.compute_micro_batch_size(args)

    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )
    summary = tensorboard.Summary(args)

    if args.use_wandb:
        utils.wandb_init(args, wandb_sweep=wandb_sweep)

    model = get_model(args, ds=train_ds, summary=summary)

    optimizer = torch.optim.AdamW(
        params=model.get_parameters(core_lr=args.core_lr_scale * args.lr),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=0,
    )
    criterion = losses.get_criterion(args, ds=train_ds)
    scaler = GradScaler(enabled=args.amp)
    if args.amp and args.verbose:
        print(f"Enable automatic mixed precision training.")
    scheduler = Scheduler(
        args, model=model, optimizer=optimizer, scaler=scaler, mode="max"
    )

    utils.save_args(args)
    epoch = scheduler.restore(load_optimizer=True, load_scheduler=True)

    utils.plot_samples(args, model=model, ds=val_ds, summary=summary, epoch=epoch)

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
            scaler=scaler,
            epoch=epoch,
            summary=summary,
        )
        val_result = validate(
            args,
            ds=val_ds,
            model=model,
            criterion=criterion,
            scaler=scaler,
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
            )
        if epoch % 10 == 0:
            utils.plot_samples(
                args, model=model, ds=val_ds, summary=summary, epoch=epoch
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
                    "elapse": elapse,
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
        args, model=model, ds=val_ds, summary=summary, epoch=epoch, mode=2
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
        required=True,
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mouse_ids",
        nargs="+",
        type=str,
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
        type=float,
        default=1.0,
        help="crop the center of the image to (scale * height, scale, width)",
    )
    parser.add_argument(
        "--resize_image",
        type=int,
        default=1,
        choices=[0, 1],
        help="resize image mode:"
        "0: no resizing, return full image (1, 144, 256)"
        "1: resize image to (1, 36, 64)",
    )
    parser.add_argument(
        "--gray_scale", action="store_true", help="convert colored image to gray-scale"
    )
    parser.add_argument(
        "--limit_data",
        type=int,
        default=None,
        help="limit the number of training samples.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of works for DataLoader.",
    )

    # training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="maximum epochs to train the model.",
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=0,
        help="micro batch size to train the model. if the model is being "
        "trained on CUDA device and micro batch size 0 is provided, then "
        "automatically increase micro batch size until OOM.",
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
    parser.add_argument(
        "--amp",
        action="store_true",
        help="automatic mixed precision training",
    )
    parser.add_argument(
        "--grad_checkpointing",
        type=int,
        default=None,
        choices=[0, 1],
        help="Enable gradient checkpointing if 1. If None is provided, then "
        "enable by default if CUDA is detected.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use deterministic algorithms in PyTorch",
    )

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
        type=int,
        default=1,
        choices=[0, 1],
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
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2, 3])

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
    match temp_args.core:
        case "conv":
            parser.add_argument("--num_layers", type=int, default=4)
            parser.add_argument("--num_filters", type=int, default=8)
            parser.add_argument("--dropout", type=float, default=0.0)
            parser.add_argument("--core_reg_scale", type=float, default=0)
            parser.add_argument("--lr", type=float, default=0.001)
        case "stacked2d":
            parser.add_argument("--num_layers", type=int, default=4)
            parser.add_argument("--dropout", type=float, default=0.0)
            parser.add_argument("--core_reg_input", type=float, default=6.3831)
            parser.add_argument("--core_reg_hidden", type=float, default=0.0)
            parser.add_argument(
                "--linear", action="store_true", help="remove non-linearity in core"
            )
            parser.add_argument("--lr", type=float, default=0.009)
        case "vit":
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
                "--p_dropout",
                type=float,
                default=0.0229,
                help="patch embeddings dropout",
            )
            parser.add_argument(
                "--t_dropout", type=float, default=0.2544, help="ViT block dropout"
            )
            parser.add_argument(
                "--drop_path",
                type=float,
                default=0.0,
                help="stochastic depth dropout rate",
            )
            parser.add_argument(
                "--use_lsa", action="store_true", help="Use Locality Self Attention"
            )
            parser.add_argument(
                "--disable_bias",
                action="store_true",
                help="Disable bias terms in linear layers in ViT.",
            )
            parser.add_argument("--core_reg_scale", type=float, default=0.5379)
            parser.add_argument("--lr", type=float, default=0.001647)
        case "cct":
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
            parser.add_argument("--mlp_dim", type=float, default=488)
            parser.add_argument(
                "--pos_emb", type=str, default="sine", choices=["sine", "learn", "none"]
            )
            parser.add_argument(
                "--p_dropout",
                type=float,
                default=0.0229,
                help="patch embeddings dropout",
            )
            parser.add_argument(
                "--t_dropout", type=float, default=0.2544, help="ViT block dropout"
            )
            parser.add_argument(
                "--drop_path",
                type=float,
                default=0.0,
                help="stochastic depth dropout rate",
            )
            parser.add_argument("--core_reg_scale", type=float, default=0.5379)
            parser.add_argument("--lr", type=float, default=0.001647)
        case "stn":
            parser.add_argument("--num_layers", type=int, default=7)
            parser.add_argument("--num_filters", type=int, default=63)
            parser.add_argument("--dropout", type=float, default=0.1135)
            parser.add_argument("--core_reg_scale", type=float, default=0.0450)
            parser.add_argument("--lr", type=float, default=0.001)
        case _:
            raise NotImplementedError(f"--core {temp_args.core} not implemented.")

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
