import os
import torch
import pickle
import argparse
import numpy as np
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
    mouse_id: str,
    ds: DataLoader,
    model: Model,
    hide_pupil_x: bool = False,
    hide_pupil_y: bool = False,
):
    recorder = Recorder(vit=model.core)
    i_transform_image = ds.dataset.i_transform_image
    i_transform_pupil_center = ds.dataset.i_transform_pupil_center
    i_transform_behavior = ds.dataset.i_transform_behavior
    results = {"images": [], "heatmaps": [], "behaviors": [], "pupil_centers": []}
    for batch in tqdm(ds, desc=f"Mouse {mouse_id}"):
        pupil_center = batch["pupil_center"]
        if hide_pupil_x:
            pupil_center[:, 0] = 0.0
        if hide_pupil_y:
            pupil_center[:, 1] = 0.0
        behavior = batch["behavior"]
        image, _ = model.image_cropper(
            inputs=batch["image"],
            mouse_id=mouse_id,
            pupil_centers=pupil_center,
            behaviors=behavior,
        )
        _, attention = recorder(
            images=image,
            behaviors=behavior,
            pupil_centers=pupil_center,
            mouse_id=mouse_id,
        )
        image = i_transform_image(image)
        image, attention = image.numpy()[0], attention.numpy()[0]
        heatmap = attention_rollout(image=image, attention=attention)
        results["images"].append(image)
        results["heatmaps"].append(heatmap)
        results["behaviors"].append(i_transform_behavior(behavior))
        results["pupil_centers"].append(i_transform_pupil_center(pupil_center))
        recorder.clear()
    recorder.eject()
    del recorder
    results = {k: np.stack(v, axis=0) for k, v in results.items()}
    return results


def computer_centers(heatmaps: np.ndarray):
    centers = np.zeros((len(heatmaps), 2), dtype=np.float32)
    for i, heatmap in enumerate(heatmaps):
        y, x = center_of_mass(heatmap)
        centers[i, 0], centers[i, 1] = x, y
    mid_point = np.array([64 / 2, 36 / 2])
    centers = centers - mid_point
    return centers


def mutual_information(x: np.ndarray, y: np.ndarray):
    c_xy = np.histogram2d(x, y, len(x))[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    utils.set_random_seed(1234)

    args.batch_size = 1
    args.device = torch.device(args.device)
    utils.load_args(args)

    filename = os.path.join(args.output_dir, "center_mass.pkl")
    if not os.path.exists(filename) or args.overwrite:
        _, _, test_ds = data.get_training_ds(
            args,
            data_dir=args.dataset,
            mouse_ids=args.mouse_ids,
            batch_size=args.batch_size,
            device=args.device,
        )

        model = Model(args, ds=test_ds)
        model.train(False)

        scheduler = Scheduler(args, model=model, save_optimizer=False)
        scheduler.restore(force=True)

        results = {}
        for mouse_id, mouse_ds in test_ds.items():
            if mouse_id in ("S0", "S1"):
                continue
            results[mouse_id] = extract_attention_maps(
                mouse_id=mouse_id, ds=mouse_ds, model=model
            )

        with open(filename, "wb") as file:
            pickle.dump(results, file)
    else:
        print(f"load attention maps from {filename}.")
        with open(filename, "rb") as file:
            results = pickle.load(file)

    spreads = {"x": [], "y": []}
    for mouse_id, mouse_dict in results.items():
        # compute correlation center of mass and pupil center
        mass_centers = computer_centers(mouse_dict["heatmaps"])
        pupil_centers = mouse_dict["pupil_centers"][:, 0, :]
        corr_x, p_x = pearsonr(mass_centers[:, 0], pupil_centers[:, 0])
        corr_y, p_y = pearsonr(mass_centers[:, 1], pupil_centers[:, 1])
        # standard deviation in x and y axes
        spread_x = np.std(np.sum(mouse_dict["heatmaps"], axis=1), axis=1)
        spread_y = np.std(np.sum(mouse_dict["heatmaps"], axis=2), axis=1)
        dilation = mouse_dict["behaviors"][:, 0, 0]
        # absolute correlation between pupil dilation and attention map
        # standard deviation
        corr_dx, p_dx = pearsonr(spread_x, dilation)
        corr_dy, p_dy = pearsonr(spread_y, dilation)
        spreads["x"].append(np.abs(corr_dx))
        spreads["y"].append(np.abs(corr_dy))

        print(
            f"Mouse {mouse_id}\n"
            f"\tCorr(center of mass,  pupil center)\n"
            f"\t\tx-axis: {corr_x:.03f} (p-value: {p_x:.03e})\n"
            f"\t\ty-axis: {corr_y:.03f} (p-value: {p_y:.03e})\n"
            f"\tCorr(attention map std, pupil dilation)\n"
            f"\t\tx-axis: {corr_dx:.03f} (p-value: {p_dx:.03e})\n"
            f"\t\ty-axis: {corr_dy:.03f} (p-value: {p_dy:.03e})\n"
        )

    print(
        f'mean abs corr(dilation, spread): x-axis: {np.mean(spreads["x"]):.3f}, '
        f'y-axis: {np.mean(spreads["y"]):.3f}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    main(parser.parse_args())
