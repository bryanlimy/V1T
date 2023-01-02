import os
import torch
import argparse
import typing as t
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from sensorium import data


def main(args):
    dataset, metadata = data.load_mouse_data(
        mouse_dir=os.path.join(args.dataset, "static20457-5-9-preproc0")
    )
    for tier in np.unique(metadata["tiers"]):
        print(f'{tier} size: {np.count_nonzero(metadata["tiers"] == tier)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/lurz2020")

    main(parser.parse_args())
