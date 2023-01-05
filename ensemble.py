import os
import torch
import wandb
import argparse
import typing as t
from torch import nn
from time import time
from shutil import rmtree
from einops import rearrange
from datetime import datetime
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


import submission
import train as trainer
from sensorium import losses, data
from sensorium.models.utils import ELU1
from sensorium.utils.logger import Logger
from sensorium.utils import utils, tensorboard
from sensorium.utils.scheduler import Scheduler
from sensorium.models import Model, get_model_info


class Args:
    def __init__(self, args, output_dir: str):
        self.device = args.device
        self.output_dir = output_dir


class OutputModule(nn.Module):
    """
    ensemble mode:
        0 - average the outputs of the ensemble models
        1 - linear layer to connect the outputs from the ensemble models
        2 - separate linear layer per animal
    """

    def __init__(self, args, in_features: int):
        super(OutputModule, self).__init__()
        self.in_features = in_features
        self.output_shapes = args.output_shapes
        self.ensemble_mode = args.ensemble_mode
        assert self.ensemble_mode in (0, 1, 2)
        if self.ensemble_mode == 1:
            self.linear = nn.Linear(in_features=in_features, out_features=1)
        elif self.ensemble_mode == 2:
            self.linear = nn.ModuleDict(
                {
                    str(mouse_id): nn.Linear(in_features=in_features, out_features=1)
                    for mouse_id in self.output_shapes.keys()
                }
            )
        self.activation = ELU1()

        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs: torch.Tensor, mouse_id: int):
        if self.ensemble_mode == 0:
            outputs = torch.mean(inputs, dim=-1)
        elif self.ensemble_mode == 1:
            outputs = self.linear(inputs)
            outputs = rearrange(outputs, "b d 1 -> b d")
        elif self.ensemble_mode == 2:
            outputs = self.linear[str(mouse_id)](inputs)
            outputs = rearrange(outputs, "b d 1 -> b d")
        else:
            raise NotImplementedError("--ensemble_model must be 0 or 1.")
        outputs = self.activation(outputs)
        return outputs


class EnsembleModel(nn.Module):
    def __init__(
        self,
        args,
        saved_models: t.Dict[str, str],
        ds: t.Dict[int, DataLoader],
    ):
        super(EnsembleModel, self).__init__()
        self.verbose = args.verbose
        self.input_shape = args.input_shape
        self.output_shapes = args.output_shapes
        self.ensemble = nn.ModuleDict()
        for name, output_dir in saved_models.items():
            model_args = Args(args, output_dir)
            utils.load_args(model_args)
            model = Model(args=model_args, ds=ds)
            self.load_model_state(model, output_dir=model_args.output_dir)
            self.ensemble[name] = model
        self.ensemble.requires_grad_(False)
        self.output_module = OutputModule(args, in_features=len(saved_models))

    def load_model_state(
        self,
        model: nn.Module,
        output_dir: str,
        device: torch.device = torch.device("cpu"),
    ):
        filename = os.path.join(output_dir, "ckpt", "model_state.pt")
        assert os.path.exists(filename), f"Cannot find {filename}."
        ckpt = torch.load(filename, map_location=device)
        # it is possible that the checkpoint only contains part of a model
        # hence we update the current state_dict of the model instead of
        # directly calling model.load_state_dict(ckpt['model'])
        state_dict = model.state_dict()
        state_dict.update(ckpt["model"])
        model.load_state_dict(state_dict)
        if self.verbose:
            print(
                f"Loaded checkpoint from {output_dir} "
                f"(correlation: {ckpt['value']:.04f})."
            )

    def regularizer(self, mouse_id: int):
        return torch.tensor(0.0)

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: int,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        ensemble = []
        for name in self.ensemble.keys():
            outputs, _, _ = self.ensemble[name](
                inputs,
                mouse_id=mouse_id,
                behaviors=behaviors,
                pupil_centers=pupil_centers,
                activate=False,
            )
            outputs = rearrange(outputs, "b d -> b d 1")
            ensemble.append(outputs)
        ensemble = torch.cat(ensemble, dim=-1)
        ensemble = self.output_module(ensemble, mouse_id=mouse_id)
        return ensemble, None, None  # match output signature of Model


def fit_ensemble(
    args,
    model: EnsembleModel,
    optimizer: torch.optim.Optimizer,
    criterion: losses.Loss,
    scaler: GradScaler,
    scheduler: Scheduler,
    train_ds: t.Dict[int, DataLoader],
    val_ds: t.Dict[int, DataLoader],
    test_ds: t.Dict[int, DataLoader],
):
    summary = tensorboard.Summary(args)

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
            scaler=scaler,
            epoch=epoch,
            summary=summary,
        )
        val_result = trainer.validate(
            args,
            ds=val_ds,
            model=model,
            criterion=criterion,
            scaler=scaler,
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
                f'Train\t\t\tloss: {train_result["loss"]:.04f}\t\t'
                f'correlation: {train_result["single_trial_correlation"]:.04f}\n'
                f'Validation\t\tloss: {val_result["loss"]:.04f}\t\t'
                f'correlation: {val_result["single_trial_correlation"]:.04f}\n'
                f"Elapse: {elapse:.02f}s"
            )
        early_stop = scheduler.step(val_result["single_trial_correlation"], epoch=epoch)
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": train_result["loss"],
                    "train_corr": train_result["single_trial_correlation"],
                    "val_loss": val_result["loss"],
                    "val_corr": val_result["single_trial_correlation"],
                    "best_corr": scheduler.best_value,
                    "elapse": elapse,
                },
                step=epoch,
            )
        if early_stop:
            break

    scheduler.restore()
    eval_result = utils.evaluate(
        args,
        ds=test_ds,
        model=model,
        epoch=epoch,
        summary=summary,
        mode=2,
        print_result=True,
        save_result=args.output_dir,
    )
    if args.use_wandb:
        wandb.log({"test_corr": eval_result["single_trial_correlation"]}, step=epoch)


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    Logger(args)
    utils.get_device(args)
    utils.set_random_seed(seed=args.seed)

    if not args.mouse_ids:
        args.mouse_ids = list(range(1 if args.behavior_mode else 0, 7))

    args.micro_batch_size = args.batch_size
    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.use_wandb:
        os.environ["WANDB_SILENT"] = "true"
        try:
            wandb.init(
                config=args,
                dir=os.path.join(args.output_dir, "wandb"),
                project="sensorium",
                entity="bryanlimy",
                group=args.wandb_group,
                name=os.path.basename(args.output_dir),
            )
        except AssertionError as e:
            print(f"wandb.init error: {e}\n")
            args.use_wandb = False

    # pretrained model to load
    args.saved_models = {
        "vit-1": "runs/vit_ensemble/001_vit_gaussian2d_seed1",
        "vit-2": "runs/vit_ensemble/002_vit_gaussian2d_seed2",
        "vit-3": "runs/vit_ensemble/003_vit_gaussian2d_seed3",
        "vit-4": "runs/vit_ensemble/004_vit_gaussian2d_seed4",
        "vit-5": "runs/vit_ensemble/005_vit_gaussian2d_seed5",
    }

    model = EnsembleModel(args, saved_models=args.saved_models, ds=train_ds)

    # get model info
    mouse_id = list(args.output_shapes.keys())[0]
    batch_size = args.micro_batch_size
    random_input = lambda size: torch.rand(*size)
    model_info = get_model_info(
        model=model,
        input_data=[
            random_input((batch_size, *model.input_shape)),  # image
            mouse_id,  # mouse ID
            random_input((batch_size, 3)),  # behaviors
            random_input((batch_size, 2)),  # pupil centers
        ],
        filename=os.path.join(args.output_dir, "model.txt"),
    )
    if args.verbose > 2:
        print(str(model_info))
    if args.use_wandb:
        wandb.log({"trainable_params": model_info.trainable_params}, step=0)

    model.to(args.device)

    utils.save_args(args)

    if args.ensemble_mode == 0 and args.train:
        print(f"Cannot train ensemble model with average outputs")

    criterion = losses.get_criterion(args, ds=train_ds)
    scaler = GradScaler(enabled=args.amp)
    if args.amp and args.verbose:
        print(f"Enable automatic mixed precision training.")
    if args.ensemble_mode:
        optimizer = torch.optim.AdamW(
            params=[
                {
                    "params": model.parameters(),
                    "lr": args.lr,
                    "name": "model",
                }
            ],
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        scheduler = Scheduler(
            args,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            mode="max",
            module_names=["output_module"],
        )
        if args.train:
            fit_ensemble(
                args,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                scheduler=scheduler,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
            )
        else:
            scheduler.restore()
    else:
        epoch = 0
        val_result = trainer.validate(
            args,
            ds=val_ds,
            model=model,
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
        )
        if args.verbose:
            print(
                f'Validation\t\tloss: {val_result["loss"]:.04f}\t\t'
                f'correlation: {val_result["single_trial_correlation"]:.04f}\n'
            )
        if args.use_wandb:
            wandb.log(
                {
                    "val_loss": val_result["loss"],
                    "val_corr": val_result["single_trial_correlation"],
                    "best_corr": val_result["single_trial_correlation"],
                },
                step=epoch,
            )

    test_ds, final_test_ds = data.get_submission_ds(
        args,
        data_dir=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    # create CSV dir to save results with timestamp Year-Month-Day-Hour-Minute
    timestamp = f"{datetime.now():%Y-%m-%d-%Hh%Mm}"
    csv_dir = os.path.join(args.output_dir, "submissions", timestamp)

    eval_result = utils.evaluate(
        args, ds=test_ds, model=model, print_result=True, save_result=csv_dir
    )
    if args.use_wandb:
        wandb.log({"test_corr": eval_result["single_trial_correlation"]}, step=0)

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
    # dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sensorium",
        help="path to directory where the compressed dataset is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mouse_ids",
        nargs="+",
        type=int,
        default=None,
        help="Mouse to use for training.",
    )
    parser.add_argument(
        "--behavior_mode",
        required=True,
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="behavior mode:"
        "0: do not include behavior"
        "1: concat behavior with natural image"
        "2: add latent behavior variables to each ViT block"
        "3: add latent behavior + pupil centers to each ViT block"
        "4: separate BehaviorMLP for each animal",
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="number of works for DataLoader.",
    )

    # training settings
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="maximum epochs to train the model.",
    )
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
    parser.add_argument(
        "--amp", action="store_true", help="automatic mixed precision training"
    )
    parser.add_argument(
        "--ensemble_mode",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="ensemble method: "
        "0 - average the outputs of the ensemble models, "
        "1 - linear layer to connect the outputs from the ensemble models"
        "2 - separate linear layer per animal",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="train ensemble model before inference.",
    )

    # optimizer settings
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.9999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="L2 weight decay coefficient",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="poisson",
        help="criterion (loss function) to use.",
    )
    parser.add_argument(
        "--ds_scale",
        action="store_true",
        help="scale loss by the size of the dataset",
    )

    # plot settings
    parser.add_argument(
        "--save_plots", action="store_true", help="save plots to --output_dir"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="matplotlib figure DPI",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="svg",
        choices=["pdf", "svg", "png"],
        help="file format when --save_plots",
    )

    # wandb settings
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_group", type=str, default="")

    # misc
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])

    main(parser.parse_args())
