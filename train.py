import os
import torch
import argparse
from tqdm import tqdm
from time import time
from shutil import rmtree

from sensorium.data import data
from sensorium.utils import utils, tensorboard, metrics
from sensorium.models.registry import get_model


def train_step(data, model, optimizer, loss_function):
    result = {}
    optimizer.zero_grad()
    outputs = model(data["image"])
    loss = loss_function(data["response"], outputs)
    loss.backward()
    optimizer.step()
    result["loss/loss"] = loss
    return result


def train(
    args, ds, model, optimizer, loss_function, epoch: int, summary: tensorboard.Summary
):
    results = {}
    for data in tqdm(ds, desc="Train", disable=args.verbose == 0):
        data = {k: v.to(args.device) for k, v in data.items()}
        result = train_step(
            data=data, model=model, optimizer=optimizer, loss_function=loss_function
        )
        utils.update_dict(results, result)
    for k, v in results.items():
        results[k] = torch.stack(v).mean()
        summary.scalar(k, results[k], step=epoch, mode=0)
    return results


def validation_step(data, model, loss_function):
    result = {}
    outputs = model(data["image"])
    loss = loss_function(data["response"], outputs)
    result["loss/loss"] = loss
    return result


def validate(args, ds, model, loss_function, epoch: int, summary: tensorboard.Summary):
    results = {}
    for data in tqdm(ds, desc="Validation", disable=args.verbose == 0):
        data = {k: v.to(args.device) for k, v in data.items()}
        result = validation_step(data=data, model=model, loss_function=loss_function)
        utils.update_dict(results, result)
    for k, v in results.items():
        results[k] = torch.stack(v).mean()
        summary.scalar(k, results[k], step=epoch, mode=0)
    return results


def evaluate(args, ds, model, epoch: int, summary: tensorboard.Summary):
    trial_correlations = metrics.single_trial_correlations(
        ds=ds, model=model, device=args.device
    )
    summary.plot_correlation(
        "metrics/single_trial_correlation",
        data=utils.metrics2df(trial_correlations),
        step=epoch,
        mode=1,
    )


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    utils.set_random_seed(args.seed)

    args.device = utils.get_available_device(args.no_acceleration)

    train_ds, val_ds, test_ds = data.get_data_loaders(
        args, data_dir=args.dataset, batch_size=args.batch_size, device=args.device
    )

    model = get_model(args)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    summary = tensorboard.Summary(args)

    evaluate(
        args,
        ds=val_ds,
        model=model,
        epoch=0,
        summary=summary,
    )

    epoch = 0
    while (epoch := epoch + 1) < args.epochs + 1:
        if args.verbose:
            print(f"Epoch {epoch:03d}/{args.epochs:03d}")

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

        if args.verbose:
            print(
                f'Train\t\t\tloss: {train_results["loss/loss"]:.02f}\n'
                f'Validation\t\tloss: {val_results["loss/loss"]:.02f}\n'
                f"Elapse: {elapse:.02f}s\n"
            )

    summary.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset settings
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # model settings
    parser.add_argument("--model", type=str, default="linear")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--normalization", type=str, default="instancenorm")
    parser.add_argument("--dropout", type=float, default=0.0)

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
        default="pdf",
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
