import os
import torch
import argparse
import torchinfo
import typing as t
import numpy as np
import pandas as pd
from torch import nn
from time import time
from tqdm import tqdm
from shutil import rmtree
from datetime import datetime
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from collections import OrderedDict
from sensorium import losses, data
from sensorium.models import Model
from sensorium.utils import utils, tensorboard
from sensorium.utils.scheduler import Scheduler

import submission
import train as trainer


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


class OutputModule(nn.Module):
    def __init__(self, args, num_models: int, ds: t.Dict[int, DataLoader]):
        super(OutputModule, self).__init__()
        self.mixer = args.mixer
        self.num_models = num_models
        self.ds_stats = {
            mouse_id: ds[mouse_id].dataset.response_stats for mouse_id in ds.keys()
        }
        self.output_shapes = args.output_shapes
        self.bias_mode = args.bias_mode
        self.initialize_mixer()
        self.initialize_bias()
        self.activation = ELU1()

    def response_stats(self, mouse_id: int):
        return self.ds_stats[mouse_id]

    def initialize_mixer(self):
        # assume input is in shape (batch_size, num. neurons, num. ensemble)
        if self.mixer == "dense":
            self.layer = nn.Sequential(
                nn.Linear(in_features=self.num_models, out_features=1, bias=False),
                Rearrange("b n 1 -> b n"),
            )
        elif self.mixer == "conv":
            self.layer = nn.Sequential(
                Rearrange("b n c -> b c n"),
                nn.Conv1d(
                    in_channels=self.num_models,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=False,
                ),
                Rearrange("b 1 n -> b n"),
            )

    def initialize_bias(self):
        self.biases = nn.ParameterDict({})
        for mouse_id in self.ds_stats.keys():
            stats = self.response_stats(mouse_id=mouse_id)
            if self.bias_mode == 0:
                bias = torch.zeros(size=self.output_shapes[mouse_id])
            elif self.bias_mode == 1:
                bias = torch.from_numpy(stats["mean"])
            elif self.bias_mode == 2:
                bias = torch.from_numpy(stats["mean"] / stats["std"])
            else:
                raise NotImplementedError(
                    f"bias mode {self.bias_mode} has not been implemented."
                )
            self.biases[str(mouse_id)] = nn.Parameter(bias)

    def forward(self, inputs: torch.Tensor, mouse_id: int):
        outputs = self.layer(inputs)
        outputs = outputs + self.biases[str(mouse_id)]
        outputs = self.activation(outputs)
        return outputs


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
        self.output_module = OutputModule(args, num_models=len(saved_models), ds=ds)

    def regularizer(self, mouse_id: int):
        return torch.tensor(0)

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
        outputs = self.output_module(outputs, mouse_id=mouse_id)
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

    args.saved_models = {
        "stacked2d": "runs/sensorium/077_stacked2d_gaussian2d_1cm/",
        "vit": "runs/sensorium/071_vit_4ps_gaussian2d_0.25si/",
        "stn": "runs/tuner/stn/output_dir/97fd64d2/",
    }
    model = EnsembleModel(
        args,
        saved_models=args.saved_models,
        ds=train_ds,
    )
    model.to(args.device)

    if args.train:
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
            args,
            mode="max",
            model=model,
            optimizer=optimizer,
            save_modules=["output_module"],
        )
        criterion = losses.get_criterion(args, ds=train_ds)

        utils.save_args(args)

        epoch = scheduler.restore()

        utils.evaluate(args, ds=val_ds, model=model, epoch=0, summary=summary, mode=1)

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

            eval_result = utils.evaluate(
                args,
                ds=test_ds,
                model=model,
                epoch=epoch,
                summary=summary,
                mode=2,
            )

            if scheduler.step(eval_result["single_trial_correlation"], epoch=epoch):
                break

        scheduler.restore()
    else:
        filename = os.path.join(args.output_dir, "ckpt", "best_model.pt")
        ckpt = torch.load(filename, map_location=model.device)
        ckpt_dict = ckpt["model_state_dict"]
        model_dict = model.state_dict()
        model_dict.update({f"output_module.{k}": v for k, v in ckpt_dict.items()})
        model.load_state_dict(model_dict)

    # create CSV dir to save results with timestamp Year-Month-Day-Hour-Minute
    timestamp = f"{datetime.now():%Y-%m-%d-%Hh%Mm}"
    csv_dir = os.path.join(args.output_dir, "submissions", timestamp)

    test_ds, final_test_ds = data.get_submission_ds(
        args,
        data_dir=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    utils.evaluate(
        args, ds=test_ds, model=model, print_result=True, save_result=csv_dir
    )

    # Sensorium challenge
    if 0 in test_ds:
        submission.generate_submission(
            args,
            mouse_id=0,
            test_ds=test_ds,
            final_test_ds=final_test_ds,
            model=model,
            csv_dir=os.path.join(csv_dir, "sensorium"),
        )

    # Sensorium+ challenge
    if 1 in test_ds:
        submission.generate_submission(
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
    parser.add_argument("--mixer", default="dense", type=str, choices=["dense", "conv"])

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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train", action="store_true")

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
