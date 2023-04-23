import os
import torch
import pickle
import argparse
import typing as t

from v1t import data
from v1t.utils import utils
from v1t.models.model import Model
from v1t.utils.scheduler import Scheduler
from v1t.utils.attention_rollout import extract_attention_maps

from torch.utils.data import DataLoader


def extract(ds: t.Dict[str, DataLoader], model: Model, device: torch.device = "cpu"):
    results = {}
    for mouse_id, mouse_ds in ds.items():
        if mouse_id in ("S0", "S1"):
            continue
        results[mouse_id] = extract_attention_maps(
            ds=mouse_ds, model=model, device=device
        )
    return results


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    utils.get_device(args)
    utils.set_random_seed(1234)

    utils.load_args(args)

    _, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.dataset,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = Model(args, ds=val_ds)
    model.train(False)

    scheduler = Scheduler(args, model=model, save_optimizer=False)
    scheduler.restore(force=True)

    results = {}
    print(f"Extract attention rollout maps from validation set.")
    results["val"] = extract(ds=val_ds, model=model, device=args.device)
    print(f"\nExtract attention rollout maps from test set.")
    results["test"] = extract(ds=val_ds, model=model, device=args.device)

    filename = os.path.join(args.output_dir, "attention_rollout_maps.pkl")
    with open(filename, "wb") as file:
        pickle.dump(results, file)
    print(f"Saved attention maps to {filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)

    main(parser.parse_args())
