import os
import torch
import argparse
import typing as t
import pandas as pd
from torch import nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from sensorium.utils import utils
from sensorium.models import get_model
from sensorium.data import get_submission_ds
from sensorium.utils.checkpoint import Checkpoint


def save_csv(filename: str, results: t.Dict[str, t.List[t.Union[float, int]]]):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    df = pd.DataFrame(
        {
            "trial_indices": results["trial_ids"],
            "image_ids": results["frame_ids"],
            "responses": results["predictions"],
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
            - frame_ids: t.List[t.List[int]], frame (image) ID of the responses
            - trial_ids: t.List[t.List[str]], trial ID of the responses
            - neuron_ids: t.List[t.List[int]], neuron IDs of the responses
    """
    results = {
        "predictions": [],
        "frame_ids": [],
        "trial_ids": [],
    }
    model.train(False)
    model.requires_grad_(False)
    for data in tqdm(ds, desc=desc, disable=args.verbose == 0):
        images = data["image"].to(device)
        predictions = model(images, mouse_id=mouse_id)
        results["predictions"].extend(predictions.detach().cpu().numpy().tolist())
        results["frame_ids"].extend(data["frame_id"].numpy().tolist())
        results["trial_ids"].extend(data["trial_id"])
    # create neuron IDs for each prediction
    results["neuron_ids"] = [list(range(1, ds.dataset.num_neurons + 1))] * len(
        results["predictions"]
    )
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


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    utils.load_args(args)

    assert (
        0 in args.output_shapes and 1 in args.output_shapes
    ), "The saved model was not trained on Mouse 1 and 2."

    utils.get_device(args)

    test_ds, final_test_ds = get_submission_ds(
        args,
        data_dir=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = get_model(args, ds=test_ds)

    checkpoint = Checkpoint(args, model=model)
    checkpoint.restore(force=True)

    # run evaluation on test set for all mouse
    eval_results = utils.evaluate(args, ds=test_ds, model=model)
    print(f"Single trial correlation")
    print(
        *[
            f"Mouse {mouse_id}: {result:.04f}\t"
            for mouse_id, result in eval_results["trial_correlation"].items()
        ]
    )

    # create CSV dir to save results with timestamp Year-Month-Day-Hour-Minute
    timestamp = f"{datetime.now():%Y-%m-%d-%Hh%Mm}"
    csv_dir = os.path.join(args.output_dir, "submissions", timestamp)

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
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])

    params = parser.parse_args()
    main(params)
