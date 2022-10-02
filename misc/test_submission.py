import pandas as pd
import numpy as np
import ast


def load_csv(filename: str):
    df = pd.read_csv(filename)
    trial_idx = df["trial_indices"].values
    image_ids = df["image_ids"].values
    neuron_ids = np.array(ast.literal_eval(df["neuron_ids"].values[0]))
    predictions = np.array([ast.literal_eval(v) for v in df["prediction"].values])
    return {
        "trial_idx": trial_idx,
        "image_ids": image_ids,
        "neuron_ids": neuron_ids,
        "predictions": predictions,
    }


def main():
    filename1 = "/Users/bryanlimy/Git/sinzlab/notebooks/submission_tutorial/submission_files/submission_file_live_test.csv"
    filename2 = "/Users/bryanlimy/Git/sensorium/runs/024_vit_gaussian2d_rmsse_concurrent/submissions/2022-10-02-17h36m/sensorium/live_test.csv"

    data1 = load_csv(filename1)
    data2 = load_csv(filename2)

    print("done")


if __name__ == "__main__":
    main()
