import os
import torch
import numpy as np
import typing as t
from glob import glob
from tqdm import tqdm
from zipfile import ZipFile

from sensorium.utils import utils

# key - mouse ID, value - filename of the recordings
# Mouse 1: Sensorium, Mouse 2:Sensorium+, Mouse 3-7: pre-training
MICE = {
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
    if not os.path.exists(filename):
        raise FileNotFoundError(f"file {filename} not found.")
    with ZipFile(filename, mode="r") as file:
        file.extractall(unzip_dir)


def get_num_trials(mouse_dir: str):
    """Get the number of trials in the given mouse directory"""
    return len(glob(os.path.join(mouse_dir, "data", "images", "*.npy")))


def load_trial_data(mouse_dir: str, trial: int):
    """Load data from a single trial in mouse_dir"""
    filename, data_dir = f"{trial}.npy", os.path.join(mouse_dir, "data")
    return {
        "image": np.load(os.path.join(data_dir, "images", filename)),
        "response": np.load(os.path.join(data_dir, "responses", filename)),
        "behavior": np.load(os.path.join(data_dir, "behavior", filename)),
        "pupil_center": np.load(os.path.join(data_dir, "pupil_center", filename)),
    }


def load_mouse_metadata(mouse_dir: str):
    """Load the relevant metadata of a specific mouse
    mouse_dir: str
        - path to the mouse data directory
    coordinates: np.ndarray
        - (x, y, z) coordinates of each neuron in the cortex
    frame_id: int,
        - the unique ID for each image being shown
    tiers: str
        - 'train' and 'validation': data with labels
        - 'test': live test set
        - 'final_test': final test set, exist for mouse 1 and 2
    trial_id: int,
        - the unique ID for each trial, hidden for mouse 1 and 2
    statistics: t.Dict[str, t.Dict[str, np.ndarray]]
        - the statistics (min, max, median, mean, std) for the data
    """
    meta_dir = os.path.join(mouse_dir, "meta")
    neuron_dir = os.path.join(meta_dir, "neurons")
    trial_dir = os.path.join(meta_dir, "trials")
    stats_dir = os.path.join(meta_dir, "statistics")

    load_neuron = lambda a: np.load(os.path.join(neuron_dir, a))
    load_trial = lambda a: np.load(os.path.join(trial_dir, a))
    load_stat = lambda a, b: np.load(os.path.join(stats_dir, a, "all", f"{b}.npy"))

    stat_keys = ["min", "max", "median", "mean", "std"]
    return {
        "mouse_dir": mouse_dir,
        "num_neurons": len(load_neuron("unit_ids.npy")),
        "coordinates": load_neuron("cell_motor_coordinates.npy").astype(np.int32),
        "frame_id": load_trial("frame_image_id.npy").astype(np.int32),
        "tiers": load_trial("tiers.npy"),
        "trial_id": load_trial("trial_idx.npy").astype(np.int32),
        "stats": {
            "image": {k: load_stat("images", k) for k in stat_keys},
            "response": {k: load_stat("responses", k) for k in stat_keys},
            "behavior": {k: load_stat("behavior", k) for k in stat_keys},
            "pupil_center": {k: load_stat("pupil_center", k) for k in stat_keys},
        },
    }


def load_mouse_data(mouse_dir: str):
    """Load mouse data from mouse_dir.
    If mouse_dir does not exist, extract the zip file to the same folder.
    """
    if not os.path.isdir(mouse_dir):
        unzip(
            filename=f"{mouse_dir}.zip",
            unzip_dir=os.path.dirname(mouse_dir),
        )
    num_trials = get_num_trials(mouse_dir)
    # load data
    mouse_data = {"image": [], "response": [], "behavior": [], "pupil_center": []}
    for trial in range(num_trials):
        utils.update_dict(mouse_data, load_trial_data(mouse_dir, trial=trial))
    mouse_data = {
        k: np.stack(v, axis=0).astype(np.float32) for k, v in mouse_data.items()
    }
    return mouse_data, load_mouse_metadata(mouse_dir)


def load_mice_data(mice_dir: str, mouse_ids: t.List[int] = None, verbose: int = 1):
    if mouse_ids is None:
        mouse_ids = list(range(len(MICE)))
    mice_data, mice_meta = {}, {}
    for mouse_id in tqdm(mouse_ids, desc="Loading", disable=verbose == 0):
        mice_data[mouse_id], mice_meta[mouse_id] = load_mouse_data(
            mouse_dir=os.path.join(mice_dir, MICE[mouse_id])
        )
    return mice_data, mice_meta


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_mode: int, mice_meta: t.Dict):
        assert ds_mode in [0, 1, 2]
        self.mice_meta = mice_meta
        self.max_neurons = np.max(
            [self.mice_meta[i]["num_neurons"] for i in self.mice_meta.keys()]
        )

        tier = "train" if ds_mode == 0 else ("validation" if ds_mode == 1 else "test")
        # list of (mouse_id, trial) that belongs to tier
        self.mouse_trial = []
        for mouse in mice_meta.keys():
            trials = np.where(mice_meta[mouse]["tiers"] == tier)[0]
            self.mouse_trial.extend([(mouse, trial) for trial in trials])

    def __len__(self):
        return len(self.mouse_trial)

    def __getitem__(self, idx: t.Union[int, torch.Tensor]):
        (mouse_id, trial) = self.mouse_trial[idx]
        data = load_trial_data(
            mouse_dir=self.mice_meta[mouse_id]["mouse_dir"], trial=trial
        )
        # pad array with 0 to match max_neurons number of neurons
        pad_size = self.max_neurons - len(data["response"])

        data["response"] = np.pad(
            data["response"],
            pad_width=(0, pad_size),
            constant_values=0,
        )
        data["mouse_id"] = mouse_id
        data["coordinates"] = np.pad(
            self.mice_meta[mouse_id]["coordinates"],
            pad_width=[(0, pad_size), (0, 0)],
            constant_values=0,
        )
        data["frame_id"] = self.mice_meta[mouse_id]["frame_id"][trial]
        if type(data["frame_id"]) not in (int, np.int32, np.int64):
            data["frame_id"] = None
        return data


def get_data_loaders(
    data_dir: str,
    mouse_ids: t.List[int] = None,
    batch_size: int = 1,
    use_cuda: bool = False,
):
    if mouse_ids is None:
        mouse_ids = list(range(2, 7))
    if 0 in mouse_ids or 1 in mouse_ids:
        raise NotImplementedError(
            "Data loader for Mouse 1 and 2 have not been implemented."
        )
    mice_meta = {
        mouse_id: load_mouse_metadata(os.path.join(data_dir, MICE[mouse_id]))
        for mouse_id in mouse_ids
    }
    train_ds = Dataset(ds_mode=0, mice_meta=mice_meta)
    val_ds = Dataset(ds_mode=1, mice_meta=mice_meta)
    test_ds = Dataset(ds_mode=2, mice_meta=mice_meta)

    # initialize data loaders
    train_kwargs = {"batch_size": batch_size, "num_workers": 2, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "num_workers": 2, "shuffle": False}
    if use_cuda:
        cuda_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_ds = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    val_ds = torch.utils.data.DataLoader(val_ds, **test_kwargs)
    test_ds = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_data_loaders(data_dir="../../data", batch_size=4)

    count = 0
    for batch in tqdm(train_ds):
        count += 1
