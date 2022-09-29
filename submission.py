import os
import torch
import argparse
import typing as t
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from sensorium.utils import utils
from sensorium.models import get_model
from sensorium.data import get_submission_ds
from sensorium.utils.checkpoint import Checkpoint


def save_csv(filename: str, results: t.Dict[str, torch.Tensor]):
    if os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    df = pd.DataFrame(
        {
            "trial_indices": results["trial_ids"].numpy(),
            "image_ids": results["frame_ids"].numpy(),
            "responses": results["predictions"].numpy(),
            "neuron_ids": results["neuron_ids"].numpy(),
        }
    )

    df.to_csv(filename, index=False)
    print(f"Saved submission file {filename}.")


def inference(
    args,
    ds: DataLoader,
    model: torch.nn.Module,
    mouse_id: int,
    device: torch.device = torch.device("cpu"),
) -> t.Dict[str, torch.Tensor]:
    """
    Inference test and final test sets

    NOTE: the ground-truth file is storing **standardized responses**, meaning
    the responses of each neuron normalized by its own standard deviation.

    Returns:
        results: t.Dict[str, torch.Tensor]
            - predictions: torch.Tensor, predictions given images
            - frame_ids: torch.Tensor, frame (image) ID of the responses
            - trial_ids: torch.Tensor, trial ID of the responses
            - neuron_ids: torch.Tensor, neuron IDs of the responses
    """
    result = {
        "predictions": [],
        "frame_ids": [],
        "trial_ids": [],
    }
    model.train(False)
    for data in tqdm(ds, desc="Inference", disable=args.verbose == 0):
        images = data["image"].to(device)
        predictions = model(images, mouse_id)
        result["predictions"].append(predictions.detach().cpu())
        result["frame_ids"].append(data["frame_id"])
        result["trial_ids"].append(data["trial_id"])
    results = {k: torch.cat(v, dim=0) for k, v in result.items()}
    # create neuron IDs for each prediction
    results["neuron_ids"] = torch.arange(
        start=1, end=ds.dataset.num_neurons + 1, dtype=torch.int32
    ).repeat(len(results["predictions"]))
    return results


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    utils.load_args(args)
    utils.set_device(args)

    test_ds, final_test_ds = get_submission_ds(
        args,
        data_dir=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = get_model(args, ds=test_ds)

    checkpoint = Checkpoint(args, model=model)
    checkpoint.restore(force=True)

    # create CSV dir to save results
    args.csv_dir = os.path.join(args.output_dir, "submissions")

    # Sensorium challenge live test results
    test_results = inference(
        args, ds=test_ds[0], model=model, mouse_id=0, device=args.device
    )
    save_csv(
        filename=os.path.join(args.csv_dir, "sensorium", "live_test.csv"),
        results=test_results,
    )
    # Sensorium challenge final test results
    final_test_results = inference(
        args, ds=final_test_ds[0], model=model, mouse_id=0, device=args.device
    )
    save_csv(
        filename=os.path.join(args.csv_dir, "sensorium", "final_test.csv"),
        results=final_test_results,
    )

    print(f"\nSubmission results saved to {args.csv_dir}.")


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
