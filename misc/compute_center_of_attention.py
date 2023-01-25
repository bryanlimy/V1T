import os
import math
import torch
import pickle
import argparse
import numpy as np
import typing as t
from torch import nn
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from skimage.transform import resize
from torch.utils.data import DataLoader

from scipy.ndimage import center_of_mass

from sensorium import data
from sensorium.models.model import Model
from sensorium.utils.scheduler import Scheduler
from sensorium.utils import utils, tensorboard

from vit_visualization import Recorder, attention_rollout

utils.set_random_seed(1234)

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


from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


def mutual_information(x: np.ndarray, y: np.ndarray):
    c_xy = np.histogram2d(x, y, len(x))[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def main(args):
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")
    tensorboard.set_font()

    utils.load_args(args)
    args.batch_size = 1
    args.device = torch.device(args.device)

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

    # results = {}
    # for mouse_id, mouse_ds in test_ds.items():
    #     if mouse_id == "1":
    #         continue
    #     results[mouse_id] = extract_attention_maps(
    #         mouse_id=mouse_id, ds=mouse_ds, model=model
    #     )
    #
    # with open("center_mass.pkl", "wb") as file:
    #     pickle.dump(results, file)
    # exit()

    with open("center_mass.pkl", "rb") as file:
        results = pickle.load(file)

    for mouse_id, mouse_dict in results.items():
        # compute correlation center of mass and pupil center
        mass_centers = computer_centers(mouse_dict["heatmaps"])
        pupil_centers = mouse_dict["pupil_centers"][:, 0, :]
        corr_x, p_x = pearsonr(mass_centers[:, 0], pupil_centers[:, 0])
        corr_y, p_y = pearsonr(mass_centers[:, 1], pupil_centers[:, 1])

        mi_x = mutual_information(mass_centers[:, 0], pupil_centers[:, 0])
        mi_y = mutual_information(mass_centers[:, 1], pupil_centers[:, 1])
        print(
            f"Mouse {mouse_id}\n"
            f"\tCorr(center of mass,  pupil center)\n"
            f"\t\tx-axis: {corr_x:.03f} (p-value: {p_x:.03e}), MI: {mi_x:.04f}\n"
            f"\t\ty-axis: {corr_y:.03f} (p-value: {p_y:.03e}), MI: {mi_y:.04f}\n"
        )

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    main(parser.parse_args())
