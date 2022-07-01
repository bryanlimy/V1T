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
    """Extract zip file with filename to unzip_dir"""
    with ZipFile(filename, mode="r") as file:
        file.extractall(unzip_dir)


def get_num_trials(dir_path: str):
    """Get the number of trials in the given mouse directory"""
    return len(glob(os.path.join(dir_path, "data", "images", "*.npy")))


def load_mouse_data(mouse_dir: str) -> t.Dict[str, np.ndarray]:
    """Load mouse data from mouse_dir.
    If mouse_dir does not exist, extract the zip data to the same folder.
    """
    if not os.path.isdir(mouse_dir):
        unzip(
            filename=f"{mouse_dir}.zip",
            unzip_dir=os.path.dirname(mouse_dir),
        )
    data_dir = os.path.join(mouse_dir, "data")
    image_dir = os.path.join(data_dir, "images")
    response_dir = os.path.join(data_dir, "responses")
    behavior_dir = os.path.join(data_dir, "behavior")
    pupil_center_dir = os.path.join(data_dir, "pupil_center")

    num_trials = get_num_trials(dir_path=mouse_dir)

    mouse_data = {"image": [], "response": [], "behavior": [], "pupil_center": []}
    for trial in range(num_trials):
        filename = f"{trial}.npy"
        image = np.load(os.path.join(image_dir, filename))
        response = np.load(os.path.join(response_dir, filename))
        behavior = np.load(os.path.join(behavior_dir, filename))
        pupil_center = np.load(os.path.join(pupil_center_dir, filename))
        mouse_data["image"].append(image)
        mouse_data["response"].append(response)
        mouse_data["behavior"].append(behavior)
        mouse_data["pupil_center"].append(pupil_center)
    mouse_data = {k: np.stack(v, axis=0) for k, v in mouse_data.items()}
    return mouse_data


def load_mice_data(mice_dir: str, mouse_ids: t.List[int] = None, verbose: int = 1):
    if mouse_ids is None:
        mouse_ids = list(range(len(DATASETS)))
    mice_data = {}
    for mouse_id in tqdm(mouse_ids, desc="Loading", disable=verbose == 0):
        mice_data[mouse_id] = load_mouse_data(
            mouse_dir=os.path.join(mice_dir, DATASETS[mouse_id])
        )
    return mice_data


def load_datasets(args):
    assert os.path.isdir(args.dataset)

    data = load_mice_data(mice_dir=args.dataset)

    return data
