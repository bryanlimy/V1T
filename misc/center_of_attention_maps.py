import os
import torch
import pickle
import argparse
import numpy as np
import typing as t
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from scipy.ndimage import center_of_mass
from sklearn.metrics import mutual_info_score

from v1t import data
from v1t.utils import utils
from v1t.models.model import Model
from v1t.utils.scheduler import Scheduler


from vit_visualization import Recorder, attention_rollout


BACKGROUND_COLOR = "#ffffff"


@torch.no_grad()
def extract_attention_maps(
    ds: DataLoader, model: Model, device: torch.device = "cpu"
) -> t.Dict[str, np.ndarray]:
    model.to(device)
    model.train(False)
    mouse_id = ds.dataset.mouse_id
    i_transform_image = ds.dataset.i_transform_image
    i_transform_behavior = ds.dataset.i_transform_behavior
    i_transform_pupil_center = ds.dataset.i_transform_pupil_center
    recorder = Recorder(model.core)
    results = {"images": [], "attentions": [], "pupil_centers": [], "behaviors": []}
    for batch in tqdm(ds, desc=f"Mouse {mouse_id}"):
        images = batch["image"].to(device)
        behaviors = batch["behavior"].to(device)
        pupil_centers = batch["pupil_center"].to(device)
        images, _ = model.image_cropper(
            inputs=images,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        _, attentions = recorder(
            images=images,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
            mouse_id=mouse_id,
        )
        recorder.clear()

        results["images"].append(i_transform_image(images.cpu()))
        results["attentions"].append(attentions.cpu())
        results["behaviors"].append(i_transform_behavior(behaviors.cpu()))
        results["pupil_centers"].append(i_transform_pupil_center(pupil_centers.cpu()))

    recorder.eject()
    del recorder
    results = {k: torch.vstack(v).numpy() for k, v in results.items()}

    results["heatmaps"] = np.zeros_like(results["images"])
    for i in range(len(results["images"])):
        results["heatmaps"][i] = attention_rollout(
            image=results["images"][i], attention=results["attentions"][i]
        )
    return results


def computer_centers(heatmaps: np.ndarray):
    centers = np.zeros((len(heatmaps), 2), dtype=np.float32)
    for i, heatmap in enumerate(heatmaps):
        y, x = center_of_mass(heatmap)
        centers[i, 0], centers[i, 1] = x, y
    mid_point = np.array([64 / 2, 36 / 2])
    centers = centers - mid_point
    return centers


def abs_correlation(x: np.ndarray, y: np.ndarray) -> t.Tuple[float, str]:
    """Return the absolute Pearson correlation and its p-value in asterisk"""
    corr, p_value = pearsonr(x, y)
    asterisks = "n.s."
    if p_value <= 0.0001:
        asterisks = "****"
    elif p_value <= 0.001:
        asterisks = "***"
    elif p_value <= 0.01:
        asterisks = "**"
    elif p_value <= 0.05:
        asterisks = "*"
    return np.abs(corr), asterisks


def mutual_information(x: np.ndarray, y: np.ndarray):
    c_xy = np.histogram2d(x, y, len(x))[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    utils.get_device(args)
    utils.set_random_seed(1234)

    utils.load_args(args)

    filename = os.path.join(args.output_dir, "center_mass.pkl")
    if not os.path.exists(filename) or args.overwrite:
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
        for mouse_id, mouse_ds in test_ds.items():
            if mouse_id in ("S0", "S1"):
                continue
            results[mouse_id] = extract_attention_maps(
                ds=mouse_ds, model=model, device=args.device
            )

        with open(filename, "wb") as file:
            pickle.dump(results, file)
    else:
        print(f"load attention maps from {filename}.")
        with open(filename, "rb") as file:
            results = pickle.load(file)

    for mouse_id, mouse_dict in results.items():
        # compute correlation center of mass and pupil center
        mass_centers = computer_centers(mouse_dict["heatmaps"])
        pupil_centers = mouse_dict["pupil_centers"][:, 0, :]
        corr_x, p_x = abs_correlation(mass_centers[:, 0], pupil_centers[:, 0])
        corr_y, p_y = abs_correlation(mass_centers[:, 1], pupil_centers[:, 1])

        # standard deviation in x and y axes
        spread_x = np.std(np.sum(mouse_dict["heatmaps"], axis=1), axis=1)
        spread_y = np.std(np.sum(mouse_dict["heatmaps"], axis=2), axis=1)
        dilation = mouse_dict["behaviors"][:, 0, 0]
        # absolute correlation between pupil dilation and attention map
        # standard deviation
        corr_dx, p_dx = abs_correlation(spread_x, dilation)
        corr_dy, p_dy = abs_correlation(spread_y, dilation)

        print(
            f"Mouse {mouse_id}\n"
            f"\tAbs. Corr(center of mass,  pupil center)\n"
            f"\t\tx-axis: {corr_x:.03f} ({p_x})\n"
            f"\t\ty-axis: {corr_y:.03f} ({p_y})\n"
            f"\tAbs. Corr(attention map std, pupil dilation)\n"
            f"\t\tx-axis: {corr_dx:.03f} ({p_dx})\n"
            f"\t\ty-axis: {corr_dy:.03f} ({p_dy})\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    main(parser.parse_args())
