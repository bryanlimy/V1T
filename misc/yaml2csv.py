import os
import argparse
import pandas as pd

from v1t.utils import yaml


def main(args):
    data = yaml.load(args.input)
    df = pd.DataFrame.from_dict(data)
    df = df.transpose()
    output = args.output
    if not output:
        dirname, basename = os.path.dirname(args.input), os.path.basename(args.input)
        output = os.path.join(dirname, f"{basename.replace('.yaml','')}.csv")
    df.to_csv(output)
    print(f"saved csv to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    main(parser.parse_args())
