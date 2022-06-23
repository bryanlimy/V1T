import os
import numpy as np
from glob import glob
from tqdm import tqdm
from zipfile import ZipFile

from sensorium.utils import utils

DATASETS = {
    "sensorium": ["static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6"],
    "sensorium+": ["static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6"],
    "training": [
        "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
        "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
        "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
        "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
        "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    ],
}


def unzip(filename: str, unzip_dir: str):
    """Extract filename to unzip_dir"""
    with ZipFile(filename, mode="r") as file:
        file.extractall(unzip_dir)


def load_file(args, filename: str):

    data_dir = os.path.join(filename, "data")
    image_dir = os.path.join(data_dir, "images")
    response_dir = os.path.join(data_dir, "responses")
    behavior_dir = os.path.join(data_dir, "behavior")
    pupil_center_dir = os.path.join(data_dir, "pupil_center")

    num_trials = len(glob(os.path.join(image_dir, "*.npy")))

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


def load_set(args, set_name: str):
    assert set_name in DATASETS.keys()

    data = {}
    unzip_dir = os.path.join(args.dataset, "unzip")
    for file in tqdm(DATASETS[set_name], desc="Loading"):
        unzip_filename = os.path.join(unzip_dir, file)
        if not os.path.isdir(unzip_filename):
            unzip(
                filename=os.path.join(args.dataset, f"{file}.zip"), unzip_dir=unzip_dir
            )
        file_data = load_file(args, filename=unzip_filename)
        utils.update_dict(data, file_data)
    data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
    return data


def load_data(args):
    assert os.path.isdir(args.dataset)

    training_data = load_set(args, set_name="training")
