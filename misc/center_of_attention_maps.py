import os
import pickle
import argparse
import numpy as np
import typing as t
from scipy.stats import pearsonr
from scipy.ndimage import center_of_mass
from sklearn.metrics import mutual_info_score

from v1t import data
from v1t.utils import utils
from v1t.models.model import Model
from v1t.utils.scheduler import Scheduler
from v1t.utils.attention_rollout import extract_attention_maps


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

    center_corrs, dilation_corrs = {"x": [], "y": []}, {"x": [], "y": []}
    for mouse_id, mouse_dict in results.items():
        print(f"Mouse {mouse_id}")
        # compute correlation center of mass and pupil center
        mass_centers = computer_centers(mouse_dict["heatmaps"])
        pupil_centers = mouse_dict["pupil_centers"]
        corr_x, p_x = abs_correlation(mass_centers[:, 0], pupil_centers[:, 0])
        corr_y, p_y = abs_correlation(mass_centers[:, 1], pupil_centers[:, 1])
        center_corrs["x"].append(corr_x)
        center_corrs["y"].append(corr_y)
        print(
            f"\tAbs. Corr(center of mass, pupil center)\n"
            f"\tx-axis: {corr_x:.03f} ({p_x})\n\ty-axis: {corr_y:.03f} ({p_y})"
        )

        # standard deviation in x and y axes
        spread_x = np.std(np.sum(mouse_dict["heatmaps"], axis=1), axis=1)
        spread_y = np.std(np.sum(mouse_dict["heatmaps"], axis=2), axis=1)
        dilation = mouse_dict["behaviors"][:, 0]
        # absolute correlation between pupil dilation and attention map
        # standard deviation
        corr_x, p_x = abs_correlation(spread_x, dilation)
        corr_y, p_y = abs_correlation(spread_y, dilation)
        dilation_corrs["x"].append(corr_x)
        dilation_corrs["y"].append(corr_y)
        print(
            f"\tAbs. Corr(attention map std, pupil dilation)\n"
            f"\tx-axis: {corr_x:.03f} ({p_x})\n\ty-axis: {corr_y:.03f} ({p_y})"
        )

    print(
        f"Avg. Corr(center of mass, pupil center)\n"
        f'x-axis: {np.mean(center_corrs["x"]):.03f}\n'
        f'y-axis: {np.mean(center_corrs["y"]):.03f}\n'
        f"\nAvg. Corr(attention map std, pupil dilation)\n"
        f'x-axis: {np.mean(dilation_corrs["x"]):.03f}\n'
        f'y-axis: {np.mean(dilation_corrs["y"]):.03f}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    main(parser.parse_args())
