import os
import argparse
import sensorium


def main(args):
    data = sensorium.data.dataset.load_datasets(args)

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/raw_data",
        help="path to directory with the dataset in zip files",
    )
    main(parser.parse_args())
