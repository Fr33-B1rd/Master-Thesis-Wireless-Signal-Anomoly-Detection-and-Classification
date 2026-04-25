import argparse
import os
import pickle
import shutil

import numpy as np


def summarize_labels(name, labels):
    values, counts = np.unique(labels, return_counts=True)
    pairs = ", ".join(f"{float(v):.1f}:{int(c)}" for v, c in zip(values, counts))
    return f"{name}[{pairs}]"


def flip_binary_labels(labels, label_name, file_name):
    arr = np.asarray(labels)
    unique_values = np.unique(arr)
    allowed = {0.0, 1.0}
    if not set(float(v) for v in unique_values).issubset(allowed):
        raise ValueError(
            f"{file_name} -> {label_name} contains non-binary labels: {unique_values.tolist()}"
        )
    return 1 - arr


def process_file(path, backup_suffix):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"{os.path.basename(path)} is not a dict pickle")

    required_keys = {"train_label", "test_label"}
    missing = required_keys - data.keys()
    if missing:
        raise KeyError(f"{os.path.basename(path)} missing keys: {sorted(missing)}")

    before_train = summarize_labels("train_label", data["train_label"])
    before_test = summarize_labels("test_label", data["test_label"])

    data["train_label"] = flip_binary_labels(
        data["train_label"], "train_label", os.path.basename(path)
    ).astype(np.asarray(data["train_label"]).dtype, copy=False)
    data["test_label"] = flip_binary_labels(
        data["test_label"], "test_label", os.path.basename(path)
    ).astype(np.asarray(data["test_label"]).dtype, copy=False)

    after_train = summarize_labels("train_label", data["train_label"])
    after_test = summarize_labels("test_label", data["test_label"])

    backup_path = path + backup_suffix
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(os.path.basename(path))
    print(f"  before: {before_train}; {before_test}")
    print(f"  after : {after_train}; {after_test}")
    print(f"  backup: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Flip binary 0/1 labels in IAD dataset pickle files."
    )
    parser.add_argument("dataset_dir", help="Directory containing *_Train_Test.pkl files")
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix used for one-time backups before overwriting files",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise NotADirectoryError(dataset_dir)

    files = sorted(
        os.path.join(dataset_dir, name)
        for name in os.listdir(dataset_dir)
        if name.endswith(".pkl")
    )
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {dataset_dir}")

    for path in files:
        process_file(path, args.backup_suffix)


if __name__ == "__main__":
    main()
