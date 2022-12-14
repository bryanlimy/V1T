import os
import math
import torch
import argparse
import numpy as np
import typing as t
from torch import nn
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage.transform import resize
from torch.utils.data import DataLoader

from sensorium import data
from sensorium.models.model import Model
from sensorium.utils.scheduler import Scheduler
from sensorium.utils import utils, tensorboard


from vit_visualization import Recorder, attention_rollout

utils.set_random_seed(1234)

BACKGROUND_COLOR = "#ffffff"


@torch.no_grad()
def extract_attention_maps(
    mouse_id: int,
    ds: DataLoader,
    model: Model,
    hide_pupil_x: bool = False,
    hide_pupil_y: bool = False,
):
    recorder = Recorder(vit=model.core)
    i_transform_image = ds.dataset.i_transform_image
    results = {"images": [], "heatmaps": []}
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
        recorder.clear()
    recorder.eject()
    del recorder
    results = {k: np.stack(v, axis=0) for k, v in results.items()}
    return results


import pickle


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

    results = {}
    for mouse_id, mouse_ds in test_ds.items():
        results[mouse_id] = {}
        results[mouse_id]["hide_x"] = extract_attention_maps(
            mouse_id=mouse_id,
            ds=mouse_ds,
            model=model,
            hide_pupil_x=True,
            hide_pupil_y=False,
        )
        results[mouse_id]["hide_y"] = extract_attention_maps(
            mouse_id=mouse_id,
            ds=mouse_ds,
            model=model,
            hide_pupil_x=False,
            hide_pupil_y=True,
        )
        results[mouse_id]["hide_none"] = extract_attention_maps(
            mouse_id=mouse_id,
            ds=mouse_ds,
            model=model,
            hide_pupil_x=False,
            hide_pupil_y=False,
        )
        results[mouse_id]["hide_both"] = extract_attention_maps(
            mouse_id=mouse_id,
            ds=mouse_ds,
            model=model,
            hide_pupil_x=True,
            hide_pupil_y=True,
        )
        break

    with open("temp.pkl", "wb") as file:
        pickle.dump(results, file)

    # plot_attention_map(
    #     results=results,
    #     # filename=os.path.join(args.output_dir, "plots", "attention_rollouts.pdf"),
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/sensorium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    main(parser.parse_args())
