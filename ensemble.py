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

from sensorium import data
from sensorium.utils import utils


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


from sensorium.models.model import get_model, Model

from collections import namedtuple


class Args:
    def __init__(self, args, output_dir: str):
        self.device = args.device
        self.output_dir = output_dir


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
            temp = torch.rand(2, 1, 36, 64)
            outputs = model(temp, mouse_id=0, pupil_center=None)
        self.ensemble = nn.ModuleDict(ensemble)


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    utils.load_args(args)

    utils.get_device(args)

    test_ds, final_test_ds = data.get_submission_ds(
        args,
        data_dir=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = EnsembleModel(
        args,
        saved_models={"stacked2d": "runs/sensorium/077_stacked2d_gaussian2d_1cm/"},
        ds=test_ds,
    )

    # create CSV dir to save results with timestamp Year-Month-Day-Hour-Minute
    timestamp = f"{datetime.now():%Y-%m-%d-%Hh%Mm}"
    csv_dir = os.path.join(args.output_dir, "submissions", timestamp)

    # run evaluation on test set for all mouse
    utils.evaluate(
        args, ds=test_ds, model=model, print_result=True, save_result=csv_dir
    )

    # Sensorium challenge
    generate_submission(
        args,
        mouse_id=0,
        test_ds=test_ds,
        final_test_ds=final_test_ds,
        model=model,
        csv_dir=os.path.join(csv_dir, "sensorium"),
    )

    # Sensorium+ challenge
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
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])

    params = parser.parse_args()
    main(params)