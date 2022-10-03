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
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from einops.layers.torch import Reduce
from torchvision.datasets import ImageFolder

from sensorium.models.core import get_core
from sensorium.utils import utils, tensorboard
from sensorium.utils.checkpoint import Checkpoint


IMAGE_SIZE = (1, 144, 256)
NUM_CLASSES = 1000
IMAGE_MEAN = torch.tensor(0.44531356896770125)
IMAGE_STD = torch.tensor(0.2692461874154524)


def plot_image(
    args,
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
    num_plots: int = 5,
):
    for i in range(min(num_plots, len(images))):
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=args.dpi)
        ax.imshow(images[i][0], cmap=tensorboard.GRAY, vmin=0, vmax=1, aspect="auto")
        ax.set_title(
            f"label: {labels[i]}   prediction: {predictions[i]}", pad=3, fontsize=10
        )
        summary.figure(f"images/image{i:03d}", figure=figure, step=epoch, mode=mode)


def get_ds(args, data_dir: str, batch_size: int, device: torch.device):
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.RandomCrop(size=(IMAGE_SIZE[1:]), pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomAdjustSharpness(sharpness_factor=0.6, p=0.25),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.RandomCrop(size=(IMAGE_SIZE[1:]), pad_if_needed=True),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )

    train_data = ImageFolder(root=data_dir, transform=train_transforms)
    val_data = ImageFolder(root=data_dir, transform=test_transforms)
    test_data = ImageFolder(root=data_dir, transform=test_transforms)

    size = len(train_data)
    indexes = np.arange(size)
    np.random.shuffle(indexes)

    train_idx = indexes[: int(size * 0.7)]
    val_idx = indexes[int(size * 0.7) : int(size * 0.85)]
    test_idx = indexes[int(size * 0.85) :]

    # settings for DataLoader
    dataloader_kwargs = {"batch_size": batch_size, "num_workers": 4}
    if device.type in ["cuda", "mps"]:
        gpu_kwargs = {"prefetch_factor": 4, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)

    train_ds = data.DataLoader(
        train_data, sampler=data.SubsetRandomSampler(train_idx), **dataloader_kwargs
    )
    val_ds = data.DataLoader(
        val_data, sampler=data.SubsetRandomSampler(val_idx), **dataloader_kwargs
    )
    test_ds = data.DataLoader(
        test_data, sampler=data.SubsetRandomSampler(test_idx), **dataloader_kwargs
    )

    args.input_shape = IMAGE_SIZE
    args.output_shape = (NUM_CLASSES,)

    return train_ds, val_ds, test_ds


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.device = args.device
        self.input_shape = IMAGE_SIZE
        self.output_shape = args.output_shape

        self.add_module(
            name="core", module=get_core(args)(args, input_shape=self.input_shape)
        )

        core_shape = self.core.shape

        self.readout = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(in_features=core_shape[0], out_features=NUM_CLASSES),
            nn.LogSoftmax(dim=1),
        )

    def regularizer(self):
        """L1 regularization"""
        return sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        outputs = self.core(inputs)
        outputs = self.readout(outputs)
        return outputs


def num_correct(y_true: torch.Tensor, y_pred: torch.Tensor):
    return (y_pred == y_true).float().sum()


def train(
    args,
    ds: data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {}
    model.train(True)
    for images, labels in tqdm(ds, desc="Train", disable=args.verbose == 0):
        optimizer.zero_grad()
        images, labels = images.to(model.device), labels.to(model.device)
        outputs = model(images)
        loss = F.nll_loss(input=outputs, target=labels)
        reg_loss = model.regularizer()
        total_loss = loss + args.reg_scale * reg_loss
        total_loss.backward()
        optimizer.step()
        predictions = torch.argmax(outputs, dim=1)
        utils.update_dict(
            results,
            {
                "loss/loss": loss.detach(),
                "loss/reg_loss": reg_loss.detach(),
                "loss/total_loss": total_loss.detach(),
                "accuracy": num_correct(labels, predictions),
            },
        )
    for k, v in results.items():
        v = torch.stack(v)
        if k == "accuracy":
            results["accuracy"] = 100 * (v.sum() / len(ds.dataset))
        else:
            results[k] = v.mean()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results


def validate(
    args,
    ds: data.DataLoader,
    model: nn.Module,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    results, make_plot = {}, True
    with torch.no_grad():
        model.train(False)
        for images, labels in tqdm(ds, desc="Val", disable=args.verbose == 0):
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            loss = F.nll_loss(input=outputs, target=labels)
            reg_loss = model.regularizer()
            total_loss = loss + args.reg_scale * reg_loss
            predictions = torch.argmax(outputs, dim=1)
            utils.update_dict(
                results,
                {
                    "loss/loss": loss.detach(),
                    "loss/reg_loss": reg_loss.detach(),
                    "loss/total_loss": total_loss.detach(),
                    "accuracy": num_correct(labels, predictions),
                },
            )
            if make_plot:
                plot_image(
                    args,
                    images=images.cpu() * IMAGE_STD + IMAGE_MEAN,
                    predictions=predictions.cpu(),
                    labels=labels.cpu(),
                    summary=summary,
                    epoch=epoch,
                    mode=mode,
                )
                make_plot = False
    for k, v in results.items():
        v = torch.stack(v)
        if k == "accuracy":
            results["accuracy"] = 100 * (v.sum() / len(ds.dataset))
        else:
            results[k] = v.mean()
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
    return results


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    utils.set_random_seed(args.seed)

    utils.get_device(args)

    train_ds, val_ds, test_ds = get_ds(
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
        print(
            f'Train\t\t\tloss: {train_results["loss/total_loss"]:.04f}\t'
            f'accuracy: {train_results["accuracy"]:.04f}\n'
            f'Validation\t\tloss: {val_results["loss/total_loss"]:.04f}\t'
            f'accuracy: {val_results["accuracy"]:.04f}\n'
            f"Elapse: {elapse:.02f}s"
        )

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
        default="../data/imagenet/",
        help="path to directory where ImageNet validation set is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)

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
