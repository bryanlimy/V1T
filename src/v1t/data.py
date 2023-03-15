import os
import torch
import numpy as np
import typing as t
from glob import glob
from tqdm import tqdm
from zipfile import ZipFile
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from v1t.utils import utils

# dataset names
DS_NAMES = t.Literal["sensorium", "franke2022"]


# key - mouse ID, value - filename of the recordings
# Mouse 0: Sensorium, Mouse 1: Sensorium+, Mouse 3-7: pre-training
SENSORIUM = {
    "0": "static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    "1": "static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    "2": "static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    "3": "static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    "4": "static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    "5": "static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
    "6": "static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6",
}

FRANKE2022 = {
    "static25311-10-26": "static25311-10-26-ColorImageNet-104e446ed0128d89c639eef0abe4655b",
    "static25340-3-19": "static25340-3-19-ColorImageNet-104e446ed0128d89c639eef0abe4655b",
    "static25704-2-12": "static25704-2-12-ColorImageNet-b23ac8521543becfd382e56c657ba29b",
    "static25830-10-4": "static25830-10-4-ColorImageNet-104e446ed0128d89c639eef0abe4655b",
    "static26085-6-3": "static26085-6-3-ColorImageNet-104e446ed0128d89c639eef0abe4655b",
    "static26142-2-11": "static26142-2-11-ColorImageNet-6a21297215f4dbb802554a60c0e72877",
    "static26426-18-13": "static26426-18-13-ColorImageNet-b23ac8521543becfd382e56c657ba29b",
    "static26470-4-5": "static26470-4-5-ColorImageNet-104e446ed0128d89c639eef0abe4655b",
    "static26644-6-2": "static26644-6-2-ColorImageNet-b23ac8521543becfd382e56c657ba29b",
    "static26872-21-6": "static26872-21-6-ColorImageNet-104e446ed0128d89c639eef0abe4655b",
}


def convert_id(mouse_id: str):
    """convert mouse ID to the ID used in paper"""
    pairs = {
        "2": "A",
        "3": "B",
        "4": "C",
        "5": "D",
        "6": "E",
        "static25311-10-26": "F",
        "static25340-3-19": "G",
        "static25704-2-12": "H",
        "static25830-10-4": "I",
        "static26085-6-3": "J",
        "static26142-2-11": "K",
        "static26426-18-13": "L",
        "static26470-4-5": "M",
        "static26644-6-2": "N",
        "static26872-21-6": "O",
    }
    return pairs[mouse_id] if mouse_id in pairs else mouse_id


def get_mouse2path(ds_name: DS_NAMES):
    assert ds_name in ("sensorium", "franke2022")
    return SENSORIUM if ds_name == "sensorium" else FRANKE2022


def get_mouse_ids(args):
    """retrieve the mouse IDs when args.mouse_ids is not provided"""
    args.ds_name = os.path.basename(args.dataset)
    match args.ds_name:
        case "sensorium":
            all_animals = list(SENSORIUM.keys())
            if not args.mouse_ids:
                args.mouse_ids = all_animals
                if args.behavior_mode > 0:
                    args.mouse_ids.remove("0")
            for mouse_id in args.mouse_ids:
                assert mouse_id in all_animals
        case "franke2022":
            all_animals = list(FRANKE2022.keys())
            if not args.mouse_ids:
                args.mouse_ids = all_animals
            for mouse_id in args.mouse_ids:
                assert mouse_id in all_animals
        case _:
            raise KeyError(f"--dataset {args.ds_name} not implemented.")


class CycleDataloaders:
    """
    Cycles through dataloaders until the loader with the largest size is
    exhausted.

    Code reference: https://github.com/sinzlab/neuralpredictors/blob/9b85300ab854be1108b4bf64b0e4fa2e960760e0/neuralpredictors/training/cyclers.py#L68
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


def micro_batching(batch: t.Dict[str, torch.Tensor], batch_size: int):
    """Divide batch into micro batches"""
    indexes = np.arange(0, len(batch["image"]), step=batch_size, dtype=int)
    for i in indexes:
        yield {k: v[i : i + batch_size] for k, v in batch.items()}


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


def get_image_shape(mouse_dir: str):
    image = np.load(os.path.join(mouse_dir, "data", "images", "0.npy"))
    return image.shape


def str2datetime(array: np.ndarray):
    """Convert timestamps to datetime"""
    fn = lambda s: np.datetime64(datetime.strptime(s[11:-2], "%Y-%m-%d %H:%M:%S"))
    return np.vectorize(fn)(array)


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


def load_mouse_metadata(
    ds_name: DS_NAMES, mouse_dir: str, load_timestamps: bool = False
):
    """
    Args:
        ds_name: DS_NAMES, sensorium or franke2022
        mouse_dir: str, path to folder to host mouse recordings and metadata
        load_timestamps: bool, load timestamps from metadata
    Load the relevant metadata of a specific mouse
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
        "tiers": load_trial("tiers.npy"),
        "stats": {
            "image": {k: load_stat("images", k) for k in stat_keys},
            "response": {k: load_stat("responses", k) for k in stat_keys},
            "behavior": {k: load_stat("behavior", k) for k in stat_keys},
            "pupil_center": {k: load_stat("pupil_center", k) for k in stat_keys},
        },
    }
    # load image IDs
    if ds_name == "sensorium":
        metadata["image_ids"] = load_trial("frame_image_id.npy")
    else:
        metadata["image_ids"] = load_trial("colorframeprojector_image_id.npy")

    # load trial timestamps
    if load_timestamps:
        if ds_name == "sensorium":
            metadata["trial_ts"] = load_trial("frame_trial_ts.npy")
        else:
            metadata["trial_ts"] = load_trial("colorframeprojector_trial_ts.npy")
        metadata["trial_ts"] = str2datetime(metadata["trial_ts"])

    # load animal ID
    animal_ids = np.unique(load_neuron("animal_ids.npy"))
    assert len(animal_ids) == 1, f"Multiple animal ID in {os.path.dirname(meta_dir)}."
    metadata["animal_id"] = animal_ids[0]

    # load trial IDs
    metadata["trial_ids"] = load_trial("trial_idx.npy")
    if np.issubdtype(metadata["trial_ids"].dtype, np.integer):
        metadata["trial_ids"] = metadata["trial_ids"].astype(np.int32)
    return metadata


def load_mouse_data(ds_name: DS_NAMES, mouse_dir: str, load_timestamps: bool = False):
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
    return mouse_data, load_mouse_metadata(
        ds_name, mouse_dir=mouse_dir, load_timestamps=load_timestamps
    )


def load_mice_data(
    ds_name: DS_NAMES,
    mice_dir: str,
    mouse_ids: t.List[str] = None,
    load_timestamps: bool = False,
    verbose: int = 1,
):
    """Load data and metadata for mouse_ids into dictionaries where key is the mouse_id"""
    mouse2path = get_mouse2path(ds_name)
    if mouse_ids is None:
        mouse_ids = list(mouse2path.keys())
    mice_data, mice_meta = {}, {}
    for mouse_id in tqdm(mouse_ids, desc="Loading", disable=verbose == 0):
        mice_data[mouse_id], mice_meta[mouse_id] = load_mouse_data(
            ds_name=ds_name,
            mouse_dir=os.path.join(mice_dir, mouse2path[mouse_id]),
            load_timestamps=load_timestamps,
        )
    return mice_data, mice_meta


class MiceDataset(Dataset):
    def __init__(self, args, tier: str, data_dir: str, mouse_id: str):
        """Construct Dataset

        Note that the trial index (i.e. X in data/images/X.npy) is not the same
        as trial IDs (the numbers in meta/trials/trial_idx.npy)

        Args:
            - tier: str, train, validation, test or final_test
            - data_dir: str, path to where all data are stored
            - mouse_id: str, the mouse ID
        """
        assert tier in ("train", "validation", "test", "final_test")
        self.tier = tier
        self.mouse_id = mouse_id
        self.ds_name = args.ds_name
        assert self.ds_name in ("sensorium", "franke2022")
        mouse2path = get_mouse2path(self.ds_name)
        mouse_dir = os.path.join(data_dir, mouse2path[mouse_id])
        metadata = load_mouse_metadata(self.ds_name, mouse_dir=mouse_dir)
        self.behavior_mode = args.behavior_mode
        if self.behavior_mode and mouse_id == 0:
            raise ValueError("Mouse 0 does not have behaviour data.")
        self.mouse_dir = metadata["mouse_dir"]
        self.neuron_ids = metadata["neuron_ids"]
        self.coordinates = metadata["coordinates"]
        self.stats = metadata["stats"]
        # extract indexes that correspond to the tier
        indexes = np.where(metadata["tiers"] == tier)[0].astype(np.int32)
        if tier == "train" and hasattr(args, "limit_data") and args.limit_data:
            indexes = np.random.choice(indexes, size=args.limit_data, replace=False)
            if args.verbose:
                print(f"limit training samples to {args.limit_data}.")
        self.indexes = indexes
        self.image_ids = metadata["image_ids"][self.indexes]
        self.trial_ids = metadata["trial_ids"][self.indexes]
        # standardizer for responses
        self.compute_response_precision()

        # indicate if trial IDs and targets are hashed
        self.hashed = self.ds_name == "sensorium" and mouse_id in ("0", "1")

        self.image_shape = get_image_shape(mouse_dir)

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
        if self.behavior_mode == 1:
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

    def i_transform_pupil_center(self, pupil_center: np.ndarray):
        stats = self.pupil_stats
        return (pupil_center * stats["std"]) + stats["mean"]

    def transform_behavior(self, behavior: np.ndarray):
        """standardize behaviour"""
        stats = self.behavior_stats
        return behavior / stats["std"]

    def i_transform_behavior(self, behavior: np.ndarray):
        stats = self.behavior_stats
        return behavior * stats["std"]

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
        data["image_id"] = self.image_ids[idx]
        data["trial_id"] = self.trial_ids[idx]
        data["mouse_id"] = self.mouse_id
        return data


def get_training_ds(
    args,
    data_dir: str,
    mouse_ids: t.List[str],
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
        train_ds: t.Dict[str, DataLoader], dictionary of DataLoaders of the
            training sets where keys are the mouse IDs.
        val_ds: t.Dict[str, DataLoader], dictionary of DataLoaders of the
            validation sets where keys are the mouse IDs.
        test_ds: t.Dict[str, DataLoader], dictionary of DataLoaders of the test
            sets where keys are the mouse IDs.
    """
    if not hasattr(args, "ds_name"):
        args.ds_name = os.path.basename(args.dataset)

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
    if not hasattr(args, "ds_name"):
        args.ds_name = os.path.basename(args.dataset)
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
        if mouse_id in ("0", "1"):
            final_test_ds[mouse_id] = DataLoader(
                MiceDataset(
                    args, tier="final_test", data_dir=data_dir, mouse_id=mouse_id
                ),
                **test_kwargs,
            )

    return test_ds, final_test_ds
