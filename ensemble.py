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
from sensorium.models.utils import ELU1


class Args:
    def __init__(self, args, output_dir: str):
        self.device = args.device
        self.output_dir = output_dir


class OutputModule(nn.Module):
    def __init__(self, args, in_features: int):
        super(OutputModule, self).__init__()
        self.in_features = in_features
        self.output_shapes = args.output_shapes
        self.reg_scale = torch.tensor(args.reg_scale, device=args.device)
        # input shape (batch_size, num_neurons, num. models)
        # self.networks = nn.ModuleDict(
        #     {
        #         str(mouse_id): nn.Sequential(
        #             nn.Linear(in_features=self.in_features, out_features=1),
        #             Rearrange("b n 1 -> b n"),
        #         )
        #         for mouse_id in self.output_shapes.keys()
        #     }
        # )
        self.networks = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1),
            Rearrange("b n 1 -> b n"),
        )
        self.activation = ELU1()

    def regularizer(self, mouse_id: int):
        reg = sum(p.abs().sum() for p in self.networks.parameters())
        return self.reg_scale * reg

    def forward(self, inputs: torch.Tensor, mouse_id: int):
        outputs = self.networks(inputs)
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
            model.train(False)
            model.requires_grad_(False)
            ensemble[name] = model
        self.ensemble = nn.ModuleDict(ensemble)
        self.output_module = OutputModule(args, in_features=len(saved_models))

    def regularizer(self, mouse_id: int):
        return self.output_module.regularizer(mouse_id=mouse_id)

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
    utils.set_random_seed(seed=args.seed)

    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=None,
        batch_size=args.batch_size,
        device=args.device,
    )

    args.saved_models = {
        "stacked2d1": "runs/sensorium/077_stacked2d_gaussian2d_1cm/",
        "stacked2d2": "runs/sensorium/092_stacked2d_gaussian2d_1cm",
        # "stacked2d3": "runs/sensorium/097_stacked2d_5nl_gaussian2d",
        "vit1": "runs/sensorium/071_vit_4ps_gaussian2d_0.25si/",
        "stn1": "runs/tuner/stn/output_dir/97fd64d2/",
        # "stn2": "runs/tuner/stn/output_dir/16431fa0/",
    }
    model = EnsembleModel(
        args,
        saved_models=args.saved_models,
        ds=train_ds,
    )
    model.to(args.device)

    if args.train:
        summary = tensorboard.Summary(args)
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

        optimizer = torch.optim.Adam(
            params=[
                {
                    "params": model.parameters(),
                    "lr": args.lr,
                    "name": "model",
                }
            ],
            lr=args.lr,
        )
        scheduler = Scheduler(
            args,
            mode="max",
            model=model,
            optimizer=optimizer,
            lr_patience=2,
            min_epochs=4,
        )
        criterion = losses.get_criterion(args, ds=train_ds)

        utils.save_args(args)

        epoch = scheduler.restore()

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
        utils.save_model(args, model=model, epoch=epoch)
    else:
        filename = os.path.join(args.output_dir, "ckpt", "best_model.pt")
        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location=args.device)
            ckpt_dict = ckpt["model_state_dict"]
            model.load_state_dict(ckpt_dict)
            print(f'\nLoaded model (epoch {ckpt["epoch"]}) from {filename}.')

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
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="",
        help="Device to use for computation. "
        "Use the best available device if --device is not specified.",
    )
    parser.add_argument("--seed", type=int, default=1234)

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reg_scale", type=float, default=0)
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
