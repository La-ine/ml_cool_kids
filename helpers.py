"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def _read_csv_rows(path):
    """Read a CSV file as raw string rows (header + data), using only the
    standard library csv module (no pandas). Returns (header, rows)."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def _rows_to_float_array(header, rows, path_label=""):
    """Convert a list of raw string rows into a numpy float array, dropping
    any row whose column count does not match the header (logged via
    print). Dropped row indices are returned so matching rows can be
    dropped from a paired file (e.g. the same Ids removed from y_train
    when they are removed from x_train)."""
    n_cols_expected = len(header)
    good_mask = [len(row) == n_cols_expected for row in rows]
    n_dropped = len(rows) - sum(good_mask)
    dropped_idx = []
    if n_dropped:
        dropped_idx = [i for i, ok in enumerate(good_mask) if not ok]
        print(
            f"Warning: {path_label} -- dropping {n_dropped} malformed row(s) "
            f"at index(es) {dropped_idx} (expected {n_cols_expected} columns)."
        )
    kept_rows = [row for row, ok in zip(rows, good_mask) if ok]
    kept_rows = [["nan" if v == "" else v for v in row] for row in kept_rows]
    data = np.array(kept_rows, dtype=float)
    return data, dropped_idx


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respective numpy arrays.
    Remember to put the 3 files in the same folder and to not change the
    names of the files.

    Robust to a small number of malformed rows (column-count mismatches)
    in x_train.csv: such rows are dropped, and the matching Id is dropped
    from y_train.csv to keep both arrays aligned.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsampled.

    Returns:
        x_train, x_test, y_train, train_ids, test_ids
    """
    header_x, rows_x = _read_csv_rows(os.path.join(data_path, "x_train.csv"))
    data_x, dropped_idx = _rows_to_float_array(header_x, rows_x, "x_train.csv")
    train_ids = data_x[:, 0].astype(dtype=int)
    x_train = data_x[:, 1:]

    header_y, rows_y = _read_csv_rows(os.path.join(data_path, "y_train.csv"))
    if dropped_idx:
        dropped_ids = {rows_y[i][0] for i in dropped_idx}
        rows_y = [row for row in rows_y if row[0] not in dropped_ids]
    data_y, _ = _rows_to_float_array(header_y, rows_y, "y_train.csv")
    y_train = data_y[:, 1].astype(dtype=int)

    header_test, rows_test = _read_csv_rows(os.path.join(data_path, "x_test.csv"))
    data_test, _ = _rows_to_float_array(header_test, rows_test, "x_test.csv")
    test_ids = data_test[:, 0].astype(dtype=int)
    x_test = data_test[:, 1:]

    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required
    for a submission in Kaggle or AIcrowd.

    Args:
        ids (list, np.array): indices
        y_pred (list, np.array): predictions in {-1, 1}
        name (str): name of the file to be created
    """
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})