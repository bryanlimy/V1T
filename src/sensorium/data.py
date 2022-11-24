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


def get_image_shape(data_dir: str, mouse_id: int = 0):
    image = np.load(os.path.join(data_dir, MICE[mouse_id], "data", "images", "0.npy"))
    return image.shape


def load_trial_data(
    mouse_dir: str, trial: int, to_tensor: bool = False
) -> t.Dict[str, t.Union[np.ndarray, torch.Tensor]]:
    """Load data from a single trial in mouse_dir"""
    filename, data_dir = f"{trial}.npy", os.path.join(mouse_dir, "data")

    def _load_data(item: str):
        data = np.load(os.path.join(data_dir, item, filename)).astype(np.float32)
        return torch.from_numpy(data) if to_tensor else data

    return {
        "image": _load_data("images"),
        "response": _load_data("responses"),
        "behavior": _load_data("behavior"),
        "pupil_center": _load_data("pupil_center"),
    }


def load_mouse_metadata(mouse_dir: str):
    """Load the relevant metadata of a specific mouse
    mouse_dir: str
        - path to the mouse data directory
    coordinates: np.ndarray
        - (x, y, z) coordinates of each neuron in the cortex
    image_id: int,
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
        "neuron_ids": load_neuron("unit_ids.npy").astype(np.int32),
        "coordinates": load_neuron("cell_motor_coordinates.npy").astype(np.float32),
        "image_id": load_trial("frame_image_id.npy").astype(np.int32),
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
    def __init__(self, args, tier: str, data_dir: str, mouse_id: int):
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
        self.include_behavior = args.include_behavior
        if self.include_behavior and mouse_id == 0:
            raise ValueError("Mouse 0 does not have behaviour data.")
        self.mouse_dir = metadata["mouse_dir"]
        self.neuron_ids = metadata["neuron_ids"]
        self.coordinates = metadata["coordinates"]
        self.stats = metadata["stats"]
        # extract indexes that correspond to the tier
        self.indexes = np.where(metadata["tiers"] == tier)[0].astype(np.int32)
        self.image_ids = metadata["image_id"][self.indexes]
        self.trial_ids = metadata["trial_id"][self.indexes]
        # standardizer for responses
        self.compute_response_precision()

        # indicate if trial IDs and targets are hashed
        self.hashed = mouse_id in (0, 1)

        image_shape = get_image_shape(data_dir, mouse_id=mouse_id)
        # include the 3 behaviour data as channel of the image
        # if self.include_behaviour:
        #     image_shape = (image_shape[0] + 3, image_shape[1], image_shape[2])
        self.image_shape = image_shape

    def __len__(self):
        return len(self.indexes)

    @property
    def image_stats(self):
        return self.stats["image"]

    @property
    def response_stats(self):
        return self.stats["response"]

    @property
    def behavior_stats(self):
        return self.stats["behavior"]

    @property
    def pupil_stats(self):
        return self.stats["pupil_center"]

    @property
    def num_neurons(self):
        return len(self.neuron_ids)

    def transform_image(self, image: np.ndarray):
        stats = self.image_stats
        return (image - stats["mean"]) / stats["std"]

    def i_transform_image(self, image: t.Union[np.ndarray, torch.Tensor]):
        """Reverse standardized image"""
        if self.include_behavior:
            image = (
                torch.unsqueeze(image[0], dim=0)
                if len(image.shape) == 3
                else torch.unsqueeze(image[:, 0, :, :], dim=1)
            )
        stats = self.image_stats
        image = (image * stats["std"]) + stats["mean"]
        return image

    def transform_pupil_center(self, pupil_center: np.ndarray):
        """standardize pupil center"""
        stats = self.pupil_stats
        return (pupil_center - stats["mean"]) / stats["std"]

    def transform_behavior(self, behavior: np.ndarray):
        """standardize behaviour"""
        stats = self.behavior_stats
        return behavior / stats["std"]

    def compute_response_precision(self):
        """
        Standardize response by dividing the per neuron std if the std is
        greater than 1% of the mean std (to avoid division by 0)
        """
        std = self.response_stats["std"]
        threshold = 0.01 * np.mean(std)
        idx = std > threshold
        response_precision = np.ones_like(std) / threshold
        response_precision[idx] = 1 / std[idx]
        self._response_precision = response_precision

    def transform_response(self, response: t.Union[np.ndarray, torch.Tensor]):
        return response * self._response_precision

    def i_transform_response(self, response: t.Union[np.ndarray, torch.Tensor]):
        return response / self._response_precision

    def add_behavior_to_image(self, image: np.ndarray, behavior: np.ndarray):
        # broadcast each behavior variable to (height, width)
        behaviour = np.tile(behavior, reps=(image.shape[1], image.shape[2], 1))
        # transpose to (channel, height, width)
        behaviour = np.transpose(behaviour, axes=[2, 0, 1])
        image = np.concatenate((image, behaviour), axis=0)
        return image

    def __getitem__(self, idx: t.Union[int, torch.Tensor]):
        """Return data and metadata

        Returns
            - data, t.Dict[str, torch.Tensor]
                - image: the natural image in (C, H, W) format
                - response: the corresponding response
                - behavior: pupil size, the derivative of pupil size, and speed
                - pupil_center: the (x, y) coordinate of the center of the pupil
                - image_id: the frame image ID
                - trial_id: the trial ID
                - mouse_id: the mouse ID
        """
        trial = self.indexes[idx]
        data = load_trial_data(mouse_dir=self.mouse_dir, trial=trial)
        data["image"] = self.transform_image(data["image"])
        data["response"] = self.transform_response(data["response"])
        data["behavior"] = self.transform_behavior(data["behavior"])
        data["pupil_center"] = self.transform_pupil_center(data["pupil_center"])
        # if self.include_behaviour:
        #     data["image"] = self.add_behavior_to_image(
        #         image=data["image"], behavior=data["behavior"]
        #     )
        data["image_id"] = self.image_ids[idx]
        data["trial_id"] = self.trial_ids[idx]
        data["mouse_id"] = self.mouse_id
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
        mouse_ids = list(range(1, 7)) if args.include_behavior else list(range(0, 7))

    # settings for DataLoader
    dataloader_kwargs = {"batch_size": batch_size, "num_workers": args.num_workers}
    if device.type in ("cuda", "mps"):
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)

    # a dictionary of DataLoader for each train, validation and test set
    train_ds, val_ds, test_ds = {}, {}, {}
    args.output_shapes = {}

    for mouse_id in mouse_ids:
        train_ds[mouse_id] = DataLoader(
            MiceDataset(args, tier="train", data_dir=data_dir, mouse_id=mouse_id),
            shuffle=True,
            **dataloader_kwargs,
        )
        val_ds[mouse_id] = DataLoader(
            MiceDataset(args, tier="validation", data_dir=data_dir, mouse_id=mouse_id),
            **dataloader_kwargs,
        )
        test_ds[mouse_id] = DataLoader(
            MiceDataset(args, tier="test", data_dir=data_dir, mouse_id=mouse_id),
            **dataloader_kwargs,
        )
        args.output_shapes[mouse_id] = (train_ds[mouse_id].dataset.num_neurons,)

    args.input_shape = train_ds[mouse_ids[0]].dataset.image_shape

    return train_ds, val_ds, test_ds


def get_submission_ds(
    args,
    data_dir: str,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
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
    test_kwargs = {"batch_size": batch_size, "num_workers": 0, "shuffle": False}
    if device.type in ["cuda", "mps"]:
        test_kwargs.update({"prefetch_factor": 2, "pin_memory": True})

    # a dictionary of DataLoader for each live test and final test set
    test_ds, final_test_ds = {}, {}

    for mouse_id in list(args.output_shapes.keys()):
        test_ds[mouse_id] = DataLoader(
            MiceDataset(args, tier="test", data_dir=data_dir, mouse_id=mouse_id),
            **test_kwargs,
        )
        if mouse_id in (0, 1):
            final_test_ds[mouse_id] = DataLoader(
                MiceDataset(
                    args, tier="final_test", data_dir=data_dir, mouse_id=mouse_id
                ),
                **test_kwargs,
            )

    return test_ds, final_test_ds


class CycleDataloaders:
    """
    Cycles through dataloaders until the loader with the largest size is
    exhausted.
    """

    def __init__(self, ds: t.Dict[int, DataLoader]):
        self.ds = ds
        self.max_iterations = max([len(ds) for ds in self.ds.values()])

    @staticmethod
    def cycle(iterable: t.Iterable):
        # see https://github.com/pytorch/pytorch/issues/23900
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def __iter__(self):
        cycles = [self.cycle(loader) for loader in self.ds.values()]
        for mouse_id, mouse_ds, _ in zip(
            self.cycle(self.ds.keys()),
            (self.cycle(cycles)),
            range(len(self.ds) * self.max_iterations),
        ):
            yield mouse_id, next(mouse_ds)

    def __len__(self):
        return len(self.ds) * self.max_iterations
