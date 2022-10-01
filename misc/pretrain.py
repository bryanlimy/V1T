import os
import torch
import argparse
import typing as t
import numpy as np
from torch import nn
import torchinfo
from tqdm import tqdm
from time import time
from shutil import rmtree
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet


from sensorium import losses, metrics
from sensorium.models.core import get_core
from sensorium.data import get_training_ds
from sensorium.utils import utils, tensorboard
from sensorium.utils.checkpoint import Checkpoint


IMAGE_SIZE = (1, 144, 256)


def get_ds(args, data_dir: str, batch_size: int, device: torch.device):
    dataset = ImageNet(
        root=data_dir,
        split="train",
        transform=nn.Sequential(
            transforms.Grayscale(),
            transforms.Resize(size=IMAGE_SIZE[1]),
            transforms.RandomCrop(size=IMAGE_SIZE[1:]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ),
    )

    # settings for DataLoader
    dataloader_kwargs = {"batch_size": batch_size, "num_workers": 1}
    if device.type in ["cuda", "mps"]:
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)

    train_ds = DataLoader(dataset, **dataloader_kwargs)

    args.input_shape = IMAGE_SIZE
    args.output_shape = None

    return train_ds, None, None


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.input_shape = IMAGE_SIZE
        self.output_shape = args.output_shape

        self.add_module(
            name="core", module=get_core(args)(args, input_shape=self.input_shape)
        )

        core_shape = self.core.shape

        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=int(np.prod(core_shape)),
                out_features=self.output_shape[-1],
            ),
            nn.GELU(),
            nn.Dropout1d(p=args.dropout),
            nn.Linear(
                in_features=self.output_shape[-1], out_features=self.output_shape[-1]
            ),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor):
        outputs = self.core(inputs)
        outputs = self.readout(outputs)
        return outputs


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    return (torch.argmax(y_pred.detach(), dim=-1) == y_true.detach()).float().sum()


def train(
    args,
    ds: DataLoader,
    model: nn.Module,
    optimizer: torch.optim,
    criterion: nn.NLLLoss,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {"loss": [], "accuracy": []}
    model.train(True)
    model.requires_grad_(True)
    for data in tqdm(ds, desc="Train", disable=args.verbose == 0):
        optimizer.zero_grad()
        images = data["image"].to(model.device)
        targets = data["target"].to(model.device)
        predictions = model(images)
        loss = criterion(predictions, targets)
        results["loss"].append(loss.detach())
        results["accuracy"].append(accuracy(targets, predictions))
        loss.backward()
        optimizer.step()
    for k, v in results.items():
        results[k] = torch.stack(v).mean()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results


def validate(
    args,
    ds: DataLoader,
    model: nn.Module,
    criterion: nn.NLLLoss,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    results = {"loss": [], "accuracy": []}
    model.train(False)
    model.requires_grad_(False)
    for data in tqdm(ds, desc="Val", disable=args.verbose == 0):
        images = data["image"].to(model.device)
        targets = data["target"].to(model.device)
        predictions = model(images)
        loss = criterion(predictions, targets)
        results["loss"].append(loss.detach())
        results["accuracy"].append(accuracy(targets, predictions))
    for k, v in results.items():
        results[k] = torch.stack(v).mean()
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

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=10,
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
            f'Train\t\t\tloss: {train_results["loss"]:.04f}\t'
            f'accuracy: {train_results["accuracy"]:.04f}\n'
            f'Validation\t\tloss: {val_results["loss"]:.04f}\t'
            f'correlation: {val_results["accuracy"]:.04f}\n'
            f"Elapse: {elapse:.02f}s"
        )

        scheduler.step(val_results["loss/loss"])

        if checkpoint.monitor(loss=val_results["loss"], epoch=epoch):
            break

    checkpoint.restore()

    validate(
        args,
        ds=test_ds,
        model=model,
        criterion=criterion,
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
        default="../data/imagenet",
        help="path to directory where the compressed dataset is stored.",
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

    # misc
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])

    params = parser.parse_args()
    main(params)
