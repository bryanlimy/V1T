import os
import math
import wandb
import argparse


def main(args):
    search_space = {
        "dataset": {"value": args.dataset},
        "output_dir": {"value": os.path.join(args.output_dir, "output_dir")},
        "mouse_ids": {"value": ""},
        "behavior_mode": {"values": [1, 2, 3]}
        if args.include_behavior
        else {"value": 0},
        "center_crop": {"value": 1.0},
        "resize_image": {"value": 1},
        "num_workers": {"value": args.num_workers},
        "epochs": {"value": args.epochs},
        "batch_size": {"value": args.batch_size},
        "device": {"value": ""},
        "seed": {"value": args.seed},
        "adam_beta1": {"value": 0.9},
        "adam_beta2": {"value": 0.9999},
        "adam_eps": {"value": 1e-8},
        "criterion": {"value": "poisson"},
        "lr": {"max": 0.1, "min": 0.0001, "distribution": "uniform"},
        "ds_scale": {"value": True},
        "pretrain_core": {"value": ""},
        "core_lr_scale": {"max": 1.0, "min": 0.0001, "distribution": "uniform"},
        "save_plots": {"value": False},
        "dpi": {"value": 120},
        "format": {"value": "svg"},
        "use_wandb": {"value": True},
        "wandb_group": {"value": args.wandb_group},
        "clear_output_dir": {"value": False},
        "verbose": {"value": 1},
        "core": {"value": args.core},
        "readout": {"value": "gaussian2d"},
        "shift_mode": {"values": [0, 1, 2, 3]},
        "patch_size": {"min": 2, "max": 16, "distribution": "int_uniform"},
        "num_blocks": {"min": 1, "max": 8, "distribution": "int_uniform"},
        "emb_dim": {
            "min": math.log(8),
            "max": math.log(1024),
            "distribution": "q_log_uniform",
            "q": 2,
        },
        "mlp_dim": {
            "min": math.log(8),
            "max": math.log(1024),
            "distribution": "q_log_uniform",
            "q": 2,
        },
        "p_dropout": {"min": 0.0, "max": 0.5, "distribution": "uniform"},
        "t_dropout": {"min": 0.0, "max": 0.5, "distribution": "uniform"},
        "core_reg_scale": {"min": 0.0, "max": 1.0, "distribution": "uniform"},
        "disable_grid_predictor": {"value": False},
        "grid_predictor_dim": {"value": 2},
        "bias_mode": {"value": 0},
        "readout_reg_scale": {"value": 0.0076},
        "shifter_reg_scale": {"min": 0.0, "max": 1.0, "distribution": "uniform"},
    }

    sweep_configuration = {
        "method": "random",
        "name": "sweep-test",
        "metric": {"goal": "maximize", "name": "val_corr"},
        "parameters": search_space,
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Sensorium")

    print(f"Sweep ID: {sweep_id}")


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
        "--include_behavior",
        action="store_true",
        help="include behaviour data into input as additional channels.",
    )

    # search settings
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="number of works for DataLoader.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="number of search iterations."
    )
    parser.add_argument("--wandb_group", type=str, required=True)

    # model settings
    parser.add_argument(
        "--core", type=str, required=True, help="The core module to use."
    )

    # training settings
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument(
        "--epochs", default=400, type=int, help="maximum epochs to train the model."
    )
    parser.add_argument("--seed", type=int, default=1234)

    # misc
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])

    main(parser.parse_args())
