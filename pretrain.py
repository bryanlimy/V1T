import os
import torch
import argparse
import torchinfo
import typing as t
import numpy as np
from torch import nn
from tqdm import tqdm
from time import time
from shutil import rmtree
from einops.layers.torch import Reduce, Rearrange

from sensorium.models.core import get_core
from sensorium.utils import utils, tensorboard
from sensorium.utils.checkpoint import Checkpoint

from sensorium import pretrain

import sensorium.models.utils as model_utils


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.device = args.device
        self.input_shape = args.input_shape
        self.output_shape = args.output_shape

        self.add_module(
            name="core", module=get_core(args)(args, input_shape=self.input_shape)
        )

        core_shape = self.core.shape

        if args.mode == 0:
            self.readout = nn.Sequential(
                Reduce("b c h w -> b c", "mean"),
                nn.Linear(
                    in_features=core_shape[0],
                    out_features=args.output_shape[0],
                ),
                nn.LogSoftmax(dim=1),
            )
        else:
            latent_shape = model_utils.conv2d_shape(
                input_shape=core_shape,
                num_filters=core_shape[0],
                kernel_size=5,
                stride=2,
            )
            latent_dim = int(np.prod(latent_shape))
            # the target shape and dimension of the bottleneck layer
            target_shape = (latent_shape[0], 18, 32)
            target_dim = int(np.prod(target_shape))
            output_shape = model_utils.transpose_conv2d_shape(
                input_shape=target_shape,
                num_filters=target_shape[0],
                kernel_size=4,
                stride=2,
                padding=1,
            )
            output_shape = model_utils.transpose_conv2d_shape(
                input_shape=output_shape,
                num_filters=target_shape[0],
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self._output_shape = model_utils.transpose_conv2d_shape(
                input_shape=output_shape,
                num_filters=target_shape[0],
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self.readout = nn.Sequential(
                nn.Conv2d(
                    in_channels=core_shape[0],
                    out_channels=core_shape[0],
                    kernel_size=5,
                    stride=2,
                ),
                nn.BatchNorm2d(core_shape[0]),
                nn.GELU(),
                nn.Flatten(),  # bottleneck
                nn.Linear(in_features=latent_dim, out_features=target_dim),
                nn.GELU(),
                nn.Linear(in_features=target_dim, out_features=target_dim),
                Rearrange("b (c h w) -> b c h w", h=target_shape[1], w=target_shape[2]),
                nn.ConvTranspose2d(
                    in_channels=target_shape[0],
                    out_channels=target_shape[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=target_shape[0]),
                nn.GELU(),
                nn.ConvTranspose2d(
                    in_channels=target_shape[0],
                    out_channels=target_shape[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=target_shape[0]),
                nn.GELU(),
                nn.ConvTranspose2d(
                    in_channels=target_shape[0],
                    out_channels=target_shape[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(in_channels=target_shape[0], out_channels=1, kernel_size=1),
            )

    def regularizer(self):
        """L1 regularization"""
        return sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        outputs = self.core(inputs)
        outputs = self.readout(outputs)
        return outputs


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    utils.set_random_seed(args.seed)

    utils.get_device(args)

    train_ds, val_ds, test_ds = pretrain.data.get_ds(
        args,
        data_dir=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    summary = tensorboard.Summary(args)

    model = Model(args)
    model = model.to(args.device)

    model_info = torchinfo.summary(
        model,
        input_size=(args.batch_size, *args.input_shape),
        device=args.device,
        verbose=0,
    )
    with open(os.path.join(args.output_dir, "model.txt"), "w") as file:
        file.write(str(model_info))
    if args.verbose == 2:
        print(str(model_info))
    if summary is not None:
        summary.scalar("model/trainable_parameters", model_info.trainable_params)

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

    if args.mode == 0:
        train = pretrain.classification.train
        validate = pretrain.classification.validate
    else:
        train = pretrain.reconstruction.train
        validate = pretrain.reconstruction.validate

    while (epoch := epoch + 1) < args.epochs + 1:
        print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_results = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            summary=summary,
        )
        val_results = validate(
            args, ds=val_ds, model=model, epoch=epoch, summary=summary
        )

        elapse = time() - start

        summary.scalar("model/elapse", value=elapse, step=epoch, mode=0)
        summary.scalar(
            "model/learning_rate",
            value=optimizer.param_groups[0]["lr"],
            step=epoch,
            mode=0,
        )
        statement = f'Train\t\t\tloss: {train_results["loss/total_loss"]:.04f}\t'
        if "accuracy" in train_results:
            statement += f'accuracy: {train_results["accuracy"]:.04f}\n'
        else:
            statement += "\n"
        statement += f'Validation\t\tloss: {val_results["loss/total_loss"]:.04f}\t'
        if "accuracy" in val_results:
            statement += f'accuracy: {val_results["accuracy"]:.04f}\n'
        else:
            statement += "\n"
        statement += f"Elapse: {elapse:.02f}s"
        print(statement)

        scheduler.step(val_results["loss/total_loss"])

        if checkpoint.monitor(loss=val_results["loss/total_loss"], epoch=epoch):
            break

    checkpoint.restore()

    validate(
        args,
        ds=test_ds,
        model=model,
        summary=summary,
        epoch=epoch,
        mode=2,
    )

    summary.close()

    print(f"\nResults saved to {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/imagenet",
        help="path to directory where ImageNet validation set is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of works for DataLoader."
    )

    # model settings
    parser.add_argument(
        "--core", type=str, required=True, help="The core module to use."
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
        "--mode",
        type=int,
        required=True,
        choices=[0, 1],
        help="pretrain core module by classification or reconstruction:"
        "  0: classification"
        "  1: reconstruction",
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="maximum epochs to train the model."
    )
    parser.add_argument("--batch_size", default=64, type=int)
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
