import os
import torch
import argparse
import typing as t
from torch import nn
from tqdm import tqdm
from time import time
from shutil import rmtree
from torch.utils.data import DataLoader

from sensorium.models import get_model
from sensorium.data import get_data_loaders
from sensorium.utils import utils, tensorboard, metrics


def train_step(
    mouse_id: int,
    batch: t.Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim,
    loss_function,
):
    result = {}
    optimizer.zero_grad()
    model.core.requires_grad_(True)
    for m in model.readouts.keys():
        if m == mouse_id:
            model.readouts[m].requires_grad_(True)
        else:
            model.readouts[m].requires_grad_(False)
    outputs = model(batch["image"], mouse_id=mouse_id)
    loss = loss_function(batch["response"], outputs)
    loss.backward()
    optimizer.step()
    result["loss/loss"] = loss
    return result


def train(
    args,
    ds: t.Dict[int, DataLoader],
    model: nn.Module,
    optimizer,
    loss_function,
    epoch: int,
    summary: tensorboard.Summary,
):
    model.train(True)
    results = {}
    disable = args.verbose == 0
    for mouse_id, data in tqdm(ds.items(), desc="Train", disable=disable, position=0):
        for batch in tqdm(
            data, desc=f"Mouse {mouse_id}", disable=disable, position=1, leave=False
        ):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            result = train_step(
                mouse_id=mouse_id,
                batch=batch,
                model=model,
                optimizer=optimizer,
                loss_function=loss_function,
            )
            utils.update_dict(results, result)
    for k, v in results.items():
        results[k] = torch.stack(v).mean()
        summary.scalar(k, results[k], step=epoch, mode=0)
    return results


def validation_step(
    mouse_id: int,
    batch: t.Dict[str, torch.Tensor],
    model: nn.Module,
    loss_function,
):
    result = {}
    model.requires_grad_(False)
    outputs = model(batch["image"], mouse_id=mouse_id)
    loss = loss_function(batch["response"], outputs)
    result["loss/loss"] = loss
    return result


def validate(
    args,
    ds: t.Dict[int, DataLoader],
    model: nn.Module,
    loss_function,
    epoch: int,
    summary: tensorboard.Summary,
):
    model.train(False)
    results = {}
    disable = args.verbose == 0
    for mouse_id, data in tqdm(ds.items(), desc="Val", disable=disable, position=0):
        for batch in tqdm(
            data, desc=f"Mouse {mouse_id}", disable=disable, position=1, leave=False
        ):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            result = validation_step(
                mouse_id=mouse_id,
                batch=batch,
                model=model,
                loss_function=loss_function,
            )
            utils.update_dict(results, result)
    for k, v in results.items():
        results[k] = torch.stack(v).mean()
        summary.scalar(k, results[k], step=epoch, mode=1)
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
    results = utils.inference(args, ds=ds, model=model, device=args.device)
    trial_correlations = metrics.single_trial_correlations(results=results)
    summary.plot_correlation(
        "metrics/single_trial_correlation",
        data=utils.metrics2df(trial_correlations),
        step=epoch,
        mode=mode,
    )
    for mouse_id, correlation in trial_correlations.items():
        eval_result[f"trial_correlation/mouse{mouse_id}"] = torch.mean(correlation)
    if mode == 2:  # only test set has repeated images
        image_correlations = metrics.average_image_correlation(results=results)
        summary.plot_correlation(
            "metrics/average_image_correlation",
            data=utils.metrics2df(image_correlations),
            step=epoch,
            mode=mode,
        )
        for mouse_id, correlation in image_correlations.items():
            eval_result[f"image_correlation/mouse{mouse_id}"] = torch.mean(correlation)
        feve = metrics.feve(results=results)
        summary.plot_correlation(
            "metrics/FEVE",
            data=utils.metrics2df(feve),
            step=epoch,
            ylabel="FEVE",
            mode=mode,
        )
        for mouse_id, f_eve in feve.items():
            eval_result[f"feve/mouse{mouse_id}"] = torch.mean(f_eve)
    for k, v in eval_result.items():
        summary.scalar(tag=k, value=v, step=epoch, mode=mode)
    return eval_result


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    utils.set_random_seed(args.seed)

    args.device = utils.get_available_device(args.no_acceleration)

    train_ds, val_ds, test_ds = get_data_loaders(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    summary = tensorboard.Summary(args)

    model = get_model(args, summary=summary)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch = utils.load_checkpoint(args, model=model, optimizer=optimizer)
    evaluate(args, ds=val_ds, model=model, epoch=0, summary=summary, mode=1)

    while (epoch := epoch + 1) < args.epochs + 1:
        print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_results = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch=epoch,
            summary=summary,
        )
        val_results = validate(
            args,
            ds=val_ds,
            model=model,
            loss_function=loss_function,
            epoch=epoch,
            summary=summary,
        )
        elapse = time() - start

        summary.scalar("model/elapse", elapse, step=epoch, mode=0)

        print(
            f'Train\t\t\tloss: {train_results["loss/loss"]:.02f}\n'
            f'Validation\t\tloss: {val_results["loss/loss"]:.02f}\n'
            f"Elapse: {elapse:.02f}s\n"
        )

        if epoch % 10 == 0 or epoch == args.epochs:
            evaluate(args, ds=val_ds, model=model, epoch=epoch, summary=summary)
            utils.save_checkpoint(args, model=model, optimizer=optimizer, epoch=epoch)

    evaluate(args, ds=test_ds, model=model, epoch=epoch, summary=summary, mode=2)

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
    parser.add_argument("--core", type=str, default="linear")
    parser.add_argument("--readout", type=str, default="linear")

    # ConvCore
    parser.add_argument("--num_filters", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="gelu")

    # training settings
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="model learning rate")
    parser.add_argument(
        "--no_acceleration",
        action="store_true",
        help="disable accelerated training and train on CPU.",
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
