import os
import torch
import argparse
import torchinfo
import typing as t
import numpy as np
from torch import nn
import torchvision.io
from tqdm import tqdm
from time import time
from PIL import Image
from shutil import rmtree
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as F

from sensorium.models.core import get_core
from sensorium.utils import utils, tensorboard
from sensorium.utils.checkpoint import Checkpoint
from sensorium.models.utils import conv2d_output_shape

from glob import glob
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (1, 144, 256)
NUM_CLASSES = 1000


class ImageNet(Dataset):
    def __init__(self, filenames: np.ndarray, labels: np.ndarray):
        super(ImageNet, self).__init__()
        self._filenames = filenames
        self._labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, item: t.Union[int, torch.Tensor]):
        filename, label = str(self._filenames[item]), self._labels[item]
        image = Image.open(filename).convert("L")  # load image in grayscale
        image = transforms.ToTensor()(image)
        image = transforms.Resize(size=IMAGE_SIZE[1:])(image)
        image = transforms.Normalize([0.449], [0.226])(image)
        return {"image": image, "label": label.type(torch.LongTensor)}


def get_ds(args, data_dir: str, batch_size: int, device: torch.device):
    with open(
        os.path.join(data_dir, "ILSVRC2012_validation_ground_truth.txt"), "r"
    ) as file:
        labels = list(map(int, file.read().splitlines()))
    labels = np.array(labels, dtype=np.int32) - 1

    filenames = sorted(glob(os.path.join(data_dir, "*.JPEG")))
    filenames = np.array(filenames)

    size = len(labels)
    # shuffle train, validation test set
    indexes = np.arange(size)
    train_idx, val_idx = train_test_split(indexes, test_size=0.3, shuffle=True)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5, shuffle=True)

    # settings for DataLoader
    dataloader_kwargs = {"batch_size": batch_size, "num_workers": 4}
    if device.type in ["cuda", "mps"]:
        gpu_kwargs = {"prefetch_factor": 4, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)

    train_ds = DataLoader(
        ImageNet(filenames=filenames[train_idx], labels=labels[train_idx]),
        **dataloader_kwargs,
    )
    val_ds = DataLoader(
        ImageNet(filenames=filenames[val_idx], labels=labels[val_idx]),
        **dataloader_kwargs,
    )
    test_ds = DataLoader(
        ImageNet(filenames=filenames[test_idx], labels=labels[test_idx]),
        **dataloader_kwargs,
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

        output_shape = conv2d_output_shape(
            input_shape=core_shape, num_filters=20, kernel_size=5
        )
        output_shape = (output_shape[0], output_shape[1] // 2, output_shape[2] // 2)
        output_shape = conv2d_output_shape(
            input_shape=output_shape, num_filters=10, kernel_size=5
        )

        self.readout = nn.Sequential(
            nn.Conv2d(
                in_channels=core_shape[0],
                out_channels=20,
                kernel_size=5,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=5,
            ),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(
                in_features=int(np.prod(output_shape)),
                out_features=self.output_shape[-1],
            ),
            nn.GELU(),
            nn.Dropout1d(p=args.dropout),
            nn.Linear(
                in_features=self.output_shape[-1], out_features=self.output_shape[-1]
            ),
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
    return (torch.argmax(y_pred.detach(), dim=1) == y_true.detach()).float().sum()


def train(
    args,
    ds: DataLoader,
    model: nn.Module,
    optimizer: torch.optim,
    criterion: nn.NLLLoss,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {"loss": [], "correct": []}
    model.train(True)
    model.requires_grad_(True)
    for data in tqdm(ds, desc="Train", disable=args.verbose == 0):
        optimizer.zero_grad()
        images = data["image"].to(model.device)
        labels = data["label"].to(model.device)
        predictions = model(images)
        loss = criterion(predictions, labels)
        reg_loss = model.regularizer()
        total_loss = loss + args.reg_scale * reg_loss
        total_loss.backward()
        optimizer.step()
        utils.update_dict(
            results,
            {
                "loss": loss.detach(),
                "reg_loss": reg_loss.detach(),
                "total_loss": total_loss.detach(),
                "correct": num_correct(labels, predictions),
            },
        )
    for k, v in results.items():
        v = torch.stack(v)
        if k == "correct":
            results["accuracy"] = 100 * (v.sum() / len(ds.dataset))
            del results["correct"]
        else:
            results[k] = v.mean()
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
    results = {}
    model.train(False)
    model.requires_grad_(False)
    for data in tqdm(ds, desc="Val", disable=args.verbose == 0):
        images = data["image"].to(model.device)
        labels = data["label"].to(model.device)
        predictions = model(images)
        loss = criterion(predictions, labels)
        reg_loss = model.regularizer()
        total_loss = loss + args.reg_scale * reg_loss
        utils.update_dict(
            results,
            {
                "loss": loss.detach(),
                "reg_loss": reg_loss.detach(),
                "total_loss": total_loss.detach(),
                "correct": num_correct(labels, predictions),
            },
        )
    for k, v in results.items():
        v = torch.stack(v)
        if k == "correct":
            results["accuracy"] = 100 * (v.sum() / len(ds.dataset))
            del results["correct"]
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
            f'accuracy: {val_results["accuracy"]:.04f}\n'
            f"Elapse: {elapse:.02f}s"
        )

        scheduler.step(val_results["loss"])

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
        default="../data/imagenet/ILSVRC2012_img_val",
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
