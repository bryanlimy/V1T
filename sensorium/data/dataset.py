import os
import numpy as np
import typing as t
from glob import glob
from tqdm import tqdm
from zipfile import ZipFile

from sensorium.utils import utils

# key - mouse ID, value - filename of the recordings
# Mouse 1: Sensorium, Mouse 2:Sensorium+, Mouse 3-7: pre-training
DATASETS = {
    0: "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    1: "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    2: "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    3: "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    4: "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    5: "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    6: "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
}


def unzip(filename: str, unzip_dir: str):
    """Extract filename to unzip_dir"""
    with ZipFile(filename, mode="r") as file:
        file.extractall(unzip_dir)


def get_num_trials(dir_path: str):
    """Get the number of trials in the given mouse directory"""
    return len(glob(os.path.join(dir_path, "data", "images", "*.npy")))


def load_mouse_data(mouse_dir: str):
    data_dir = os.path.join(mouse_dir, "data")
    image_dir = os.path.join(data_dir, "images")
    response_dir = os.path.join(data_dir, "responses")
    behavior_dir = os.path.join(data_dir, "behavior")
    pupil_center_dir = os.path.join(data_dir, "pupil_center")

    num_trials = get_num_trials(dir_path=mouse_dir)

    data = {"image": [], "response": [], "behavior": [], "pupil_center": []}
    for trial in range(num_trials):
        filename = f"{trial}.npy"
        image = np.load(os.path.join(image_dir, filename))
        response = np.load(os.path.join(response_dir, filename))
        behavior = np.load(os.path.join(behavior_dir, filename))
        pupil_center = np.load(os.path.join(pupil_center_dir, filename))
        data["image"].append(image)
        data["response"].append(response)
        data["behavior"].append(behavior)
        data["pupil_center"].append(pupil_center)
    data = {k: np.stack(v, axis=0) for k, v in data.items()}
    return data


def load_mice_data(args, mouse_ids: t.List[int] = None):
    if mouse_ids is None:
        mouse_ids = list(range(len(DATASETS)))
    data = {}
    unzip_dir = os.path.join(args.dataset, "unzip")
    for mouse_id in tqdm(mouse_ids, desc="Loading"):
        mouse_dir = os.path.join(unzip_dir, DATASETS[mouse_id])
        if not os.path.isdir(mouse_dir):
            unzip(
                filename=os.path.join(args.dataset, f"{DATASETS[mouse_id]}.zip"),
                unzip_dir=unzip_dir,
            )
        data[mouse_id] = load_mouse_data(mouse_dir=mouse_dir)
    return data


def load_datasets(args):
    assert os.path.isdir(args.dataset)

    data = load_mice_data(args)

    return data
