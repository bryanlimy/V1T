import os
import torch
import argparse
import typing as t
from torch import nn
from tqdm import tqdm
from time import time
from shutil import rmtree
from torch.utils.data import DataLoader

from sensorium import losses, metrics
from sensorium.models import get_model
from sensorium.data import get_training_ds
from sensorium.utils import utils, tensorboard
from sensorium.utils.checkpoint import Checkpoint


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Metrics to compute as part of training and validation step"""
    y_true, y_pred = y_true.detach().cpu(), y_pred.detach().cpu()
    trial_corr = metrics.correlation(y1=y_true, y2=y_pred, axis=None)
    return {"metrics/trial_correlation": torch.tensor(trial_corr)}


def train_step(
    mice_data: t.Tuple[t.Dict[str, torch.Tensor]],
    model: nn.Module,
    optimizer: torch.optim,
    criterion: losses.Loss,
    reg_scale: t.Union[float, torch.Tensor] = 0,
) -> t.Dict[int, t.Dict[str, torch.Tensor]]:
    result = {}
    optimizer.zero_grad()
    all_loss = []
    for data in mice_data:
        mouse_id = int(data["mouse_id"][0])
        images = data["image"].to(model.device)
        responses = data["response"].to(model.device)
        outputs = model(images, mouse_id=mouse_id)
        loss = criterion(y_true=responses, y_pred=outputs, mouse_id=mouse_id)
        reg_loss = model.regularizer()
        total_loss = loss + reg_scale * reg_loss
        all_loss.append(total_loss)
        result[mouse_id] = {
            "loss/loss": loss.detach(),
            "loss/reg_loss": reg_loss.detach(),
            "loss/total_loss": total_loss.detach(),
            **compute_metrics(y_true=responses, y_pred=outputs),
        }
    total_loss = torch.sum(torch.stack(all_loss))
    total_loss.backward()
    optimizer.step()
    return result


def train(
    args,
    ds: t.Dict[int, DataLoader],
    model: nn.Module,
    optimizer: torch.optim,
    criterion: losses.Loss,
    epoch: int,
    summary: tensorboard.Summary,
) -> t.Dict[t.Union[str, int], t.Union[torch.Tensor, t.Dict[str, torch.Tensor]]]:
    model.train(True)
    model.requires_grad_(True)
    mouse_ids = list(ds.keys())
    results = {mouse_id: {} for mouse_id in mouse_ids}
    with tqdm(
        desc="Train", total=len(ds[mouse_ids[0]]), disable=args.verbose == 0
    ) as pbar:
        for mice_data in zip(*ds.values()):
            step_results = train_step(
                mice_data=mice_data,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                reg_scale=args.reg_scale,
            )
            for mouse_id in mouse_ids:
                utils.update_dict(results[mouse_id], step_results[mouse_id])
            pbar.update(1)
    for mouse_id in mouse_ids:
        utils.log_metrics(
            results=results[mouse_id],
            epoch=epoch,
            mode=0,
            summary=summary,
            mouse_id=mouse_id,
        )
    utils.log_metrics(results=results, epoch=epoch, mode=0, summary=summary)
    return results


def validation_step(
    mouse_id: int,
    data: t.Dict[str, torch.Tensor],
    model: nn.Module,
    criterion: losses.Loss,
) -> t.Dict[str, torch.Tensor]:
    result = {}
    images = data["image"].to(model.device)
    responses = data["response"].to(model.device)
    outputs = model(images, mouse_id=mouse_id)
    loss = criterion(y_true=responses, y_pred=outputs, mouse_id=mouse_id)
    result["loss/loss"] = loss
    result.update(compute_metrics(y_true=responses, y_pred=outputs))
    return result


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
    with tqdm(desc="Val", total=utils.num_steps(ds), disable=args.verbose == 0) as pbar:
        with torch.no_grad():
            for mouse_id, mouse_ds in ds.items():
                mouse_result = {}
                for data in mouse_ds:
                    result = validation_step(
                        mouse_id=mouse_id,
                        data=data,
                        model=model,
                        criterion=criterion,
                    )
                    utils.update_dict(mouse_result, result)
                    pbar.update(1)
                utils.log_metrics(
                    results=mouse_result,
                    epoch=epoch,
                    mode=1,
                    summary=summary,
                    mouse_id=mouse_id,
                )
                results[mouse_id] = mouse_result
    utils.log_metrics(results=results, epoch=epoch, mode=1, summary=summary)
    return results


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    utils.set_random_seed(args.seed)

    utils.get_device(args)

    train_ds, val_ds, test_ds = get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    summary = tensorboard.Summary(args)

    model = get_model(args, ds=train_ds, summary=summary)
    criterion = losses.get_criterion(args, ds=train_ds)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold_mode="rel",
        min_lr=1e-6,
        verbose=False,
    )

    utils.save_args(args)

    checkpoint = Checkpoint(args, model=model, optimizer=optimizer, scheduler=scheduler)
    epoch = checkpoint.restore()

    utils.evaluate(args, ds=val_ds, model=model, epoch=epoch, summary=summary, mode=1)

    while (epoch := epoch + 1) < args.epochs + 1:
        print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_results = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            summary=summary,
        )
        val_results = validate(
            args,
            ds=val_ds,
            model=model,
            criterion=criterion,
            epoch=epoch,
            summary=summary,
        )

        elapse = time() - start

        summary.scalar("model/elapse", value=elapse, step=epoch, mode=0)
        summary.scalar(
            "model/learning_rate",
            value=optimizer.param_groups[0]["lr"],
            step=epoch,
            mode=0,
        )
        print(
            f'Train\t\t\tloss: {train_results["loss/loss"]:.04f}\t\t'
            f'correlation: {train_results["metrics/trial_correlation"]:.04f}\n'
            f'Validation\t\tloss: {val_results["loss/loss"]:.04f}\t\t'
            f'correlation: {val_results["metrics/trial_correlation"]:.04f}\n'
            f"Elapse: {elapse:.02f}s"
        )

        scheduler.step(val_results["loss/loss"])

        if epoch % 10 == 0 or epoch == args.epochs:
            utils.evaluate(args, ds=val_ds, model=model, epoch=epoch, summary=summary)
        if checkpoint.monitor(loss=val_results["loss/loss"], epoch=epoch):
            break

    checkpoint.restore()

    utils.evaluate(args, ds=test_ds, model=model, epoch=epoch, summary=summary, mode=2)

    summary.close()

    print(f"\nResults saved to {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="data",
        help="path to directory where the compressed dataset is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mouse_ids",
        nargs="+",
        type=int,
        default=None,
        help="Mouse to use for training, use Mouse 2-7 if None.",
    )

    # model settings
    parser.add_argument(
        "--core", type=str, required=True, help="The core module to use."
    )
    parser.add_argument(
        "--readout", type=str, required=True, help="The readout module to use."
    )

    # ConvCore
    parser.add_argument("--num_filters", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="gelu")

    # ViTCore
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_head", type=int, default=64)

    # training settings
    parser.add_argument(
        "--epochs", default=200, type=int, help="maximum epochs to train the model."
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--criterion",
        default="poisson",
        type=str,
        help="criterion (loss function) to use.",
    )
    parser.add_argument(
        "--depth_scale",
        default=1.0,
        type=float,
        help="the coefficient to scale loss for neurons in depth of 240 to 260.",
    )
    parser.add_argument(
        "--reg_scale",
        default=0,
        type=float,
        help="weight regularization coefficient.",
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="model learning rate")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="",
        help="Device to use for computation. "
        "Use the best available device if --device is not specified.",
    )
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)

    # plot settings
    parser.add_argument(
        "--save_plots", action="store_true", help="save plots to --output_dir"
    )
    parser.add_argument("--dpi", type=int, default=120, help="matplotlib figure DPI")
    parser.add_argument(
        "--format",
        type=str,
        default="svg",
        choices=["pdf", "svg", "png"],
        help="file format when --save_plots",
    )

    # misc
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])

    params = parser.parse_args()
    main(params)
