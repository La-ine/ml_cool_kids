import numpy as np
import csv


def load_csv_data(x_path, y_path=None, sub_sample=False):
    """Load the raw x_train / y_train (or x_test) CSVs into numpy arrays.

    Args:
        x_path: str, path to the features CSV (e.g. "x_train.csv").
        y_path: str or None, path to the labels CSV (e.g. "y_train.csv").
                None when loading test features with no labels available.
        sub_sample: bool, if True load only every 50th row (fast local debugging).

    Returns:
        ids: numpy array of shape (N,), the "Id" column.
        x: numpy array of shape (N, D_raw), raw feature matrix (no bias column,
           columns exactly as in the CSV, still containing NaNs).
        y: numpy array of shape (N,) with entries in {-1, 1}, or None if
           y_path is None.
    """
    path_dataset_x = "./dataset/" + x_path

    with open(path_dataset_x, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    n_cols_expected = len(header)
    good_mask = [len(row) == n_cols_expected for row in rows]
    n_dropped = len(rows) - sum(good_mask)
    if n_dropped:
        dropped_idx = [i for i, ok in enumerate(good_mask) if not ok]
        print(f"Warning: dropping {n_dropped} malformed row(s) at index(es) {dropped_idx} "
              f"(column count mismatch vs header={n_cols_expected}).")
    rows = [row for row, ok in zip(rows, good_mask) if ok]

    rows = [["nan" if v == "" else v for v in row] for row in rows]
    data_x = np.array(rows, dtype=float)
    ids = data_x[:, 0]
    x = data_x[:, 1:]

    y = None
    if y_path is not None:
        path_dataset_y = "./dataset/" + y_path
        with open(path_dataset_y, "r") as f:
            reader = csv.reader(f)
            header_y = next(reader)
            rows_y = [row for row in reader]
        # keep y rows aligned with x rows: drop the SAME ids that were dropped from x
        if n_dropped:
            dropped_ids = set(rows_y[i][0] for i in dropped_idx)
            rows_y = [row for row in rows_y if row[0] not in dropped_ids]
        rows_y = [["nan" if v == "" else v for v in row] for row in rows_y]
        data_y = np.array(rows_y, dtype=float)
        y = data_y[:, 1]

    if sub_sample:
        ids = ids[::50]
        x = x[::50]
        if y is not None:
            y = y[::50]

    return ids, x, y


def clean_features(x, nan_threshold=0.8, fill_value="median"):
    """Drop columns that are mostly NaN and impute remaining missing values.

    Args:
        x: numpy array of shape (N, D_raw), possibly containing NaNs
           (e.g. _FLSHOT6, _PNEUMO2, _AIDTST3 columns in the sample head).
        nan_threshold: float in (0,1), drop any column whose fraction of
           NaN entries exceeds this threshold.
        fill_value: "median", "mean", or a float; strategy for imputing
           remaining NaNs column-wise.

    Returns:
        x_clean: numpy array of shape (N, D_kept), NaN-free.
        kept_columns: numpy array of shape (D_kept,), integer indices of the
           columns kept from the original x (needed to apply the identical
           transform to x_test later).
        fill_values: numpy array of shape (D_kept,), the imputation value
           used per kept column (needed to reproduce imputation on x_test).
    """
    nan_frac = np.isnan(x).mean(axis=0)
    kept_columns = np.where(nan_frac <= nan_threshold)[0]
    x_kept = x[:, kept_columns]

    if fill_value == "median":
        fill_values = np.nanmedian(x_kept, axis=0)
    elif fill_value == "mean":
        fill_values = np.nanmean(x_kept, axis=0)
    else:
        fill_values = np.full(x_kept.shape[1], fill_value)

    nan_mask = np.isnan(x_kept)
    x_clean = np.where(nan_mask, fill_values, x_kept)

    return x_clean, kept_columns, fill_values


def labels_to_01(y):
    """Convert labels from {-1, 1} to {0, 1} for use with cross-entropy /
    logistic_regression / reg_logistic_regression, which expect y in {0,1}.

    Args:
        y: numpy array of shape (N,), entries in {-1, 1}.

    Returns:
        numpy array of shape (N,), entries in {0, 1} (-1 -> 0, 1 -> 1).
    """
    return np.where(y == -1, 0, 1)


def labels_to_pm1(y01):
    """Inverse of labels_to_01: convert {0,1} predictions back to {-1,1}
    for submission / evaluation against the original y_train format.

    Args:
        y01: numpy array of shape (N,), entries in {0, 1}.

    Returns:
        numpy array of shape (N,), entries in {-1, 1} (0 -> -1, 1 -> 1).
    """
    return np.where(y01 == 0, -1, 1)


def standardize_columns(x, mean=None, std=None):
    """Standardize each column of x to zero mean / unit variance.

    If mean/std are None, they are computed from x (use this on x_train).
    If mean/std are given, they are applied as-is (use this on x_val/x_test
    with the TRAINING mean/std, never recomputed on val/test).

    Args:
        x: numpy array of shape (N, D).
        mean: numpy array of shape (D,) or None.
        std: numpy array of shape (D,) or None.

    Returns:
        x_std: numpy array of shape (N, D), standardized features.
        mean_x: numpy array of shape (D,), the mean used (either computed or passed in).
        std_x: numpy array of shape (D,), the std used (either computed or passed in);
             columns with std == 0 should be handled (e.g. set to 1) to avoid div-by-zero.
    """
    if mean is None:
        mean_x = np.mean(x, axis=0)
    else:
        mean_x = mean
    x = x - mean_x

    if std is None:
        std_x = np.std(x, axis=0)
        std_x = np.where(std_x == 0, 1, std_x)
    else:
        std_x = std
    x = x / std_x
    return x, mean_x, std_x


def build_tx(x):
    """Prepend the bias column of ones, matching helpers.build_model_data's
    convention (tx = [1, x_1, ..., x_D]).

    Args:
        x: numpy array of shape (N, D).

    Returns:
        tx: numpy array of shape (N, D+1).
    """
    N = x.shape[0]
    return np.c_[np.ones(N), x]