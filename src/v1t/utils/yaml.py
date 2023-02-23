import os
import torch
import numpy as np
import typing as t
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def array2py(data: t.Dict):
    """
    Recursively replace np.ndarray and torch.Tensor variables in data with
    Python integer, float or list.
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cpu().numpy().tolist()
        elif isinstance(v, torch.device):
            data[k] = v.type
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
            data[k] = float(v)
        elif isinstance(v, np.integer):
            data[k] = int(v)
        elif isinstance(v, dict):
            array2py(data[k])


def load(filename: str):
    """Load yaml file"""
    with open(filename, "r") as file:
        data = yaml.load(file)
    return data


def save(filename: str, data: t.Dict):
    """Save data dictionary to yaml file"""
    assert type(data) == dict
    array2py(data)
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(filename, "w") as file:
        yaml.dump(data, file)


def update(filename: str, data: t.Dict):
    """Update json file with filename with items in data"""
    content = {}
    if os.path.exists(filename):
        content = load(filename)
    for k, v in data.items():
        content[k] = v
    save(filename, data=content)
