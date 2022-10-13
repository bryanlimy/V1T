import os
import torch
import argparse
import typing as t
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from shutil import rmtree
from sensorium import data
from sensorium.utils import utils
import torchinfo
from time import time
from einops.layers.torch import Rearrange

from sensorium import losses, metrics, data
from sensorium.models import get_model, Model
from sensorium.utils import utils, tensorboard
from sensorium.utils.scheduler import Scheduler


import train as trainer


def save_csv(filename: str, results: t.Dict[str, t.List[t.Union[float, int]]]):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    df = pd.DataFrame(
        {
            "trial_indices": results["trial_ids"],
            "image_ids": results["image_ids"],
            "prediction": results["predictions"],
            "neuron_ids": results["neuron_ids"],
        }
    )
    df.to_csv(filename, index=False)
    print(f"Saved submission file {filename}.")


def inference(
    args,
    ds: DataLoader,
    model: nn.Module,
    mouse_id: int,
    device: torch.device = torch.device("cpu"),
    desc: str = "",
) -> t.Dict[str, t.List[t.Union[float, int]]]:
    """
    Inference test and final test sets

    NOTE: the ground-truth file is storing **standardized responses**, meaning
    the responses of each neuron normalized by its own standard deviation.

    Return:
        results: t.Dict[str, t.List[t.List[float, int or str]]
            - predictions: t.List[t.List[float]], predictions given images
            - image_ids: t.List[t.List[int]], frame (image) ID of the responses
            - trial_ids: t.List[t.List[str]], trial ID of the responses
            - neuron_ids: t.List[t.List[int]], neuron IDs of the responses
    """
    results = {
        "predictions": [],
        "image_ids": [],
        "trial_ids": [],
    }
    model.train(False)
    model.requires_grad_(False)
    for data in tqdm(ds, desc=desc, disable=args.verbose < 2):
        images = data["image"].to(device)
        pupil_center = data["pupil_center"].to(device)
        predictions = model(images, mouse_id=mouse_id, pupil_center=pupil_center)
        results["predictions"].extend(predictions.cpu().numpy().tolist())
        results["image_ids"].extend(data["image_id"].numpy().tolist())
        results["trial_ids"].extend(data["trial_id"])
    # create neuron IDs for each prediction
    results["neuron_ids"] = np.repeat(
        np.expand_dims(ds.dataset.neuron_ids, axis=0),
        repeats=len(results["predictions"]),
        axis=0,
    ).tolist()
    return results


def generate_submission(
    args,
    mouse_id: int,
    test_ds: t.Dict[int, DataLoader],
    final_test_ds: t.Dict[int, DataLoader],
    model: nn.Module,
    csv_dir: str,
):
    print(f"\nGenerate results for Mouse {mouse_id}")
    # live test results
    test_results = inference(
        args,
        ds=test_ds[mouse_id],
        model=model,
        mouse_id=mouse_id,
        device=args.device,
        desc="Live test",
    )
    save_csv(
        filename=os.path.join(csv_dir, "live_test.csv"),
        results=test_results,
    )
    # final test results
    final_test_results = inference(
        args,
        ds=final_test_ds[mouse_id],
        model=model,
        mouse_id=mouse_id,
        device=args.device,
        desc="Final test",
    )
    save_csv(
        filename=os.path.join(csv_dir, "final_test.csv"),
        results=final_test_results,
    )


class Args:
    def __init__(self, args, output_dir: str):
        self.device = args.device
        self.output_dir = output_dir


class ELU1(nn.Module):
    """ELU activation + 1 to output standardized responses"""

    def __init__(self):
        super(ELU1, self).__init__()
        self.elu = nn.ELU()

    def forward(self, inputs: torch.Tensor):
        return self.elu(inputs) + 1


class EnsembleModel(nn.Module):
    def __init__(
        self, args, saved_models: t.Dict[str, str], ds: t.Dict[int, DataLoader]
    ):
        super(EnsembleModel, self).__init__()
        self.device = args.device
        ensemble = {}
        model_args = {}
        for name, output_dir in saved_models.items():
            model_args[name] = Args(args, output_dir)
            utils.load_args(model_args[name])
            model = Model(args=model_args[name], ds=ds)
            utils.load_model_state(
                args,
                model=model,
                filename=os.path.join(
                    model_args[name].output_dir, "ckpt", "best_model.pt"
                ),
            )
            model.requires_grad_(False)
            ensemble[name] = model
        self.ensemble = nn.ModuleDict(ensemble)
        self.output_module = nn.Sequential(
            nn.Linear(in_features=len(saved_models), out_features=1),
            Rearrange("b n 1 -> b n"),
            ELU1(),
        )

    def regularizer(self, mouse_id: int):
        return 0

    def forward(self, inputs: torch.Tensor, mouse_id: int, pupil_center: torch.Tensor):
        outputs = [
            self.ensemble[name](
                inputs,
                mouse_id=mouse_id,
                pupil_center=pupil_center,
                activate=False,
            ).unsqueeze(dim=-1)
            for name in self.ensemble.keys()
        ]
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.output_module(outputs)
        return outputs


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    utils.get_device(args)

    summary = tensorboard.Summary(args)

    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=None,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = EnsembleModel(
        args,
        saved_models={
            "stacked2d": "runs/sensorium/077_stacked2d_gaussian2d_1cm/",
            "vit": "runs/sensorium/100_vit_gaussian2d_0.2dropout_convEMB/",
            "stn": "runs/tuner/stn/output_dir/97fd64d2/",
        },
        ds=train_ds,
    )
    # get model summary for the first rodent
    model_info = torchinfo.summary(
        model,
        input_size=(args.batch_size, *args.input_shape),
        device=args.device,
        verbose=0,
        mouse_id=list(args.output_shapes.keys())[0],
        pupil_center=torch.rand(size=(args.batch_size, 2)),
    )
    with open(os.path.join(args.output_dir, "model.txt"), "w") as file:
        file.write(str(model_info))
    if args.verbose == 3:
        print(str(model_info))
    if summary is not None:
        summary.scalar("model/trainable_parameters", model_info.trainable_params)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = Scheduler(
        args, mode="max", model=model.output_module, optimizer=optimizer
    )
    criterion = losses.get_criterion(args, ds=train_ds)

    utils.save_args(args)

    utils.evaluate(args, ds=val_ds, model=model, epoch=0, summary=summary, mode=1)

    epoch = 0
    while (epoch := epoch + 1) < args.epochs + 1:
        if args.verbose:
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_result = trainer.train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            summary=summary,
        )
        val_result = trainer.validate(
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
        if args.verbose:
            print(
                f'Train\t\t\tloss: {train_result["loss/loss"]:.04f}\t\t'
                f'correlation: {train_result["metrics/trial_correlation"]:.04f}\n'
                f'Validation\t\tloss: {val_result["loss/loss"]:.04f}\t\t'
                f'correlation: {val_result["metrics/trial_correlation"]:.04f}\n'
                f"Elapse: {elapse:.02f}s"
            )

        if epoch % 10 == 0 or epoch == args.epochs:
            utils.evaluate(
                args,
                ds=test_ds,
                model=model,
                epoch=epoch,
                summary=summary,
                mode=2,
            )

        if scheduler.step(val_result["metrics/trial_correlation"], epoch=epoch):
            break

    scheduler.restore()

    # create CSV dir to save results with timestamp Year-Month-Day-Hour-Minute
    timestamp = f"{datetime.now():%Y-%m-%d-%Hh%Mm}"
    csv_dir = os.path.join(args.output_dir, "submissions", timestamp)

    test_ds, final_test_ds = data.get_submission_ds(
        args,
        data_dir=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    # run evaluation on test set for all mouse
    utils.evaluate(
        args, ds=test_ds, model=model, print_result=True, save_result=csv_dir
    )

    # Sensorium challenge
    if 0 in test_ds:
        generate_submission(
            args,
            mouse_id=0,
            test_ds=test_ds,
            final_test_ds=final_test_ds,
            model=model,
            csv_dir=os.path.join(csv_dir, "sensorium"),
        )

    # Sensorium+ challenge
    if 1 in test_ds:
        generate_submission(
            args,
            mouse_id=1,
            test_ds=test_ds,
            final_test_ds=final_test_ds,
            model=model,
            csv_dir=os.path.join(csv_dir, "sensorium+"),
        )

    print(f"\nSubmission results saved to {csv_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data",
        help="path to directory where the compressed dataset is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="",
        help="Device to use for computation. "
        "Use the best available device if --device is not specified.",
    )

    parser.add_argument("--criterion", type=str, default="poisson")
    parser.add_argument("--plus", action="store_true", help="training for sensorium+.")
    parser.add_argument(
        "--num_workers", default=2, type=int, help="number of works for DataLoader."
    )
    parser.add_argument(
        "--depth_scale",
        default=1.0,
        type=float,
        help="the coefficient to scale loss for neurons in depth of 240 to 260.",
    )
    parser.add_argument(
        "--ds_scale",
        action="store_true",
        help="scale loss by the size of the dataset",
    )
    parser.add_argument(
        "--crop_mode",
        default=1,
        type=int,
        choices=[0, 1, 2, 3],
        help="image crop mode:"
        "0: no cropping and return full image (1, 144, 256)"
        "1: rescale image by 0.25 in both width and height (1, 36, 64)"
        "2: crop image based on retinotopy and rescale to (1, 36, 64)"
        "3: crop left half of the image and rotate.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)

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

    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )

    main(parser.parse_args())
