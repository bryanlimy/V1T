import os
import numpy as np
import typing as t
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def np2py(data: t.Dict):
    """Recursively replace NumPy values in data with Python integer or float"""
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
            data[k] = float(v)
        elif isinstance(v, np.integer):
            data[k] = int(v)
        elif isinstance(v, dict):
            np2py(data[k])


def load(filename: str):
    """Load yaml file"""
    with open(filename, "r") as file:
        data = yaml.load(file)
    return data


def save(filename: str, data: t.Dict):
    """Save data dictionary to yaml file"""
    assert type(data) == dict
    np2py(data)
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
