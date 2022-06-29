import os
import argparse
import matplotlib.pyplot as plt

from sensorium.data import dataset


def main(args):
    data = dataset.load_mice_data(args, mouse_ids=[2, 3, 4, 5, 6])

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/raw_data",
        help="path to directory with the dataset in zip files",
    )
    parser.add_argument("--plot_dir", type=str, default="plots")
    main(parser.parse_args())
