import os
import torch
import numpy as np
import typing as t
from glob import glob
from tqdm import tqdm
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader

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
    print(f"Unzipping {filename}...")
    with ZipFile(filename, mode="r") as file:
        file.extractall(unzip_dir)


def get_num_trials(mouse_dir: str):
    """Get the number of trials in the given mouse directory"""
    return len(glob(os.path.join(mouse_dir, "data", "images", "*.npy")))


def get_image_shape(data_dir: str):
    image = np.load(os.path.join(data_dir, MICE[2], "data", "images", "0.npy"))
    return image.shape


def load_trial_data(mouse_dir: str, trial: int):
    """Load data from a single trial in mouse_dir"""
    filename, data_dir = f"{trial}.npy", os.path.join(mouse_dir, "data")
    load_data = lambda a: np.load(os.path.join(data_dir, a, filename)).astype(
        np.float32
    )
    return {
        "image": load_data("images"),
        "response": load_data("responses"),
        "behavior": load_data("behavior"),
        "pupil_center": load_data("pupil_center"),
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
    if not os.path.isdir(mouse_dir):
        unzip(
            filename=f"{mouse_dir}.zip",
            unzip_dir=os.path.dirname(mouse_dir),
        )
    meta_dir = os.path.join(mouse_dir, "meta")
    neuron_dir = os.path.join(meta_dir, "neurons")
    trial_dir = os.path.join(meta_dir, "trials")
    stats_dir = os.path.join(meta_dir, "statistics")

    load_neuron = lambda a: np.load(os.path.join(neuron_dir, a))
    load_trial = lambda a: np.load(os.path.join(trial_dir, a))
    load_stat = lambda a, b: np.load(os.path.join(stats_dir, a, "all", f"{b}.npy"))

    stat_keys = ["min", "max", "median", "mean", "std"]
    metadata = {
        "mouse_dir": mouse_dir,
        "num_neurons": len(load_neuron("unit_ids.npy")),
        "coordinates": load_neuron("cell_motor_coordinates.npy").astype(np.int32),
        "frame_id": load_trial("frame_image_id.npy").astype(np.int32),
        "tiers": load_trial("tiers.npy"),
        "trial_id": load_trial("trial_idx.npy"),
        "stats": {
            "image": {k: load_stat("images", k) for k in stat_keys},
            "response": {k: load_stat("responses", k) for k in stat_keys},
            "behavior": {k: load_stat("behavior", k) for k in stat_keys},
            "pupil_center": {k: load_stat("pupil_center", k) for k in stat_keys},
        },
    }
    if np.issubdtype(metadata["trial_id"].dtype, np.integer):
        metadata["trial_id"] = metadata["trial_id"].astype(np.int32)
    return metadata


def load_mouse_data(mouse_dir: str):
    """Load data and metadata from mouse_dir"""
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
    """Load data and metadata for mouse_ids into dictionaries where key is the mouse_id"""
    if mouse_ids is None:
        mouse_ids = list(range(len(MICE)))
    mice_data, mice_meta = {}, {}
    for mouse_id in tqdm(mouse_ids, desc="Loading", disable=verbose == 0):
        mice_data[mouse_id], mice_meta[mouse_id] = load_mouse_data(
            mouse_dir=os.path.join(mice_dir, MICE[mouse_id])
        )
    return mice_data, mice_meta


class MiceDataset(Dataset):
    def __init__(self, tier: str, data_dir: str, mouse_id: int):
        """Construct Dataset

        Note that the trial index (i.e. X in data/images/X.npy) is not the same
        as trial IDs (the numbers in meta/trials/trial_idx.npy)

        Args:
            - tier: str, train, validation, test or final_test
            - data_dir: str, path to where all data are stored
            - mouse_id: int, the mouse ID
        """
        assert tier in ("train", "validation", "test", "final_test")
        self.tier = tier
        self.mouse_id = mouse_id
        metadata = load_mouse_metadata(os.path.join(data_dir, MICE[mouse_id]))
        self.mouse_dir = metadata["mouse_dir"]
        self.num_neurons = metadata["num_neurons"]
        self.coordinates = metadata["coordinates"]
        self.stats = metadata["stats"]
        # extract indexes that correspond to the tier
        self.indexes = np.where(metadata["tiers"] == tier)[0].astype(np.int32)
        self.frame_ids = metadata["frame_id"][self.indexes]
        self.trial_ids = metadata["trial_id"][self.indexes]
        # neurons cortical coordinates
        self.neurons_coordinate = metadata["coordinates"]
        # standardizer for responses
        self._response_precision = self.compute_response_precision()

    def __len__(self):
        return len(self.indexes)

    def transform_image(self, image: t.Union[np.ndarray, torch.Tensor]):
        """Standardize image"""
        return (image - self.stats["image"]["mean"]) / self.stats["image"]["std"]

    def i_transform_image(self, image: t.Union[np.ndarray, torch.Tensor]):
        """Reverse standardized image"""
        return (image * self.stats["image"]["std"]) + self.stats["image"]["mean"]

    def compute_response_precision(self):
        std = self.stats["response"]["std"]
        threshold = 0.01 * np.mean(std)
        idx = std > threshold
        response_precision = np.ones_like(std) / threshold
        response_precision[idx] = 1 / std[idx]
        return response_precision

    def transform_response(self, response: t.Union[np.ndarray, torch.Tensor]):
        """Standardize response by dividing the per neuron std if the std is
        greater than 1% of the mean std (to avoid division by 0)"""
        return response * self._response_precision

    def i_transform_response(self, response: t.Union[np.ndarray, torch.Tensor]):
        """Reverse standardized response"""
        return response / self._response_precision

    def transform(self, data: t.Dict[str, t.Union[torch.Tensor, np.ndarray]]):
        data["image"] = self.transform_image(data["image"])
        data["response"] = self.transform_response(data["response"])

    def i_transform(self, data: t.Dict[str, t.Union[torch.Tensor, np.ndarray]]):
        data["image"] = self.i_transform_image(data["image"])
        data["response"] = self.i_transform_response(data["response"])

    def __getitem__(self, idx: t.Union[int, torch.Tensor]):
        """Return data and metadata

        Returns
            - data, t.Dict[str, torch.Tensor]
                - image: the natural image in (C, H, W) format
                - response: the corresponding response
                - behavior: pupil size, the derivative of pupil size, and speed
                - pupil_center: the (x, y) coordinate of the center of the pupil
                - mouse_id: the mouse ID
                - num_neurons: the number of neurons in responses
                - frame_id: the frame image ID
                - trial_id: the trial ID, None if the trial ID is hidden
        """
        trial = self.indexes[idx]
        data = load_trial_data(mouse_dir=self.mouse_dir, trial=trial)
        self.transform(data)
        data["mouse_id"] = self.mouse_id
        data["frame_id"] = self.frame_ids[idx]
        data["trial_id"] = self.trial_ids[idx]
        return data


def get_training_ds(
    args,
    data_dir: str,
    mouse_ids: t.List[int] = None,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
):
    """
    Get DataLoaders for training
    Args:
        args
        data_dir: str, path to directory where the zip files are stored
        mouse_ids: t.List[int], mouse IDs to extract
        batch_size: int, batch size of the DataLoaders
        device: torch.device, the device where the data is being loaded to
    Return:
        train_ds: t.Dict[int, DataLoader], dictionary of DataLoaders of the
            training sets where keys are the mouse IDs.
        val_ds: t.Dict[int, DataLoader], dictionary of DataLoaders of the
            validation sets where keys are the mouse IDs.
        test_ds: t.Dict[int, DataLoader], dictionary of DataLoaders of the test
            sets where keys are the mouse IDs.
    """
    if mouse_ids is None:
        mouse_ids = list(range(0, 7))

    # settings for DataLoader
    train_kwargs = {"batch_size": batch_size, "num_workers": 2, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "num_workers": 2, "shuffle": False}
    if device.type in ["cuda", "mps"]:
        cuda_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # a dictionary of DataLoader for each train, validation and test set
    train_ds, val_ds, test_ds = {}, {}, {}
    args.output_shapes = {}

    for mouse_id in mouse_ids:
        train_ds[mouse_id] = DataLoader(
            MiceDataset(tier="train", data_dir=data_dir, mouse_id=mouse_id),
            **train_kwargs,
        )
        val_ds[mouse_id] = DataLoader(
            MiceDataset(tier="validation", data_dir=data_dir, mouse_id=mouse_id),
            **test_kwargs,
        )
        test_ds[mouse_id] = DataLoader(
            MiceDataset(tier="test", data_dir=data_dir, mouse_id=mouse_id),
            **test_kwargs,
        )
        args.output_shapes[mouse_id] = (train_ds[mouse_id].dataset.num_neurons,)

    args.input_shape = get_image_shape(data_dir=data_dir)

    return train_ds, val_ds, test_ds


def get_submission_ds(
    args, data_dir: str, batch_size: int, device: torch.device = torch.device("cpu")
):
    """
    Get DataLoaders for submission to Sensorium and Sensorium+
    Args:
        args
        data_dir: str, path to directory where the zip files are stored
        batch_size: int, batch size of the DataLoaders
        device: torch.device, the device where the data is being loaded to
    Return:
        test_ds: t.Dict[int, DataLoader], dictionary of DataLoaders of the
            live test set where keys are the mouse IDs
            i.e. 0 for Sensorium and 1 for Sensorium+.
        final_test_ds: t.Dict[int, DataLoader], dictionary of DataLoaders of
            the final test set where keys are the mouse IDs.
    """
    # settings for DataLoader
    test_kwargs = {"batch_size": batch_size, "num_workers": 2, "shuffle": False}
    if device.type in ["cuda", "mps"]:
        test_kwargs.update({"prefetch_factor": 2, "pin_memory": True})

    # a dictionary of DataLoader for each train, validation and test set
    test_ds, final_test_ds = {}, {}
    args.output_shapes = {}

    for mouse_id in [0, 1]:
        test_ds[mouse_id] = DataLoader(
            MiceDataset(tier="test", data_dir=data_dir, mouse_id=mouse_id),
            **test_kwargs,
        )
        final_test_ds[mouse_id] = DataLoader(
            MiceDataset(tier="final_test", data_dir=data_dir, mouse_id=mouse_id),
            **test_kwargs,
        )
        args.output_shapes[mouse_id] = (test_ds[mouse_id].dataset.num_neurons,)

    args.input_shape = get_image_shape(data_dir=data_dir)

    return test_ds, final_test_ds
