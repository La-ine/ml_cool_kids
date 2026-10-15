import numpy as np


def clean_features(x, nan_threshold=0.8, fill_value="median"):
    """Drop columns that are mostly NaN and impute remaining missing values.

    Args:
        x: numpy array of shape (N, D_raw), possibly containing NaNs.
        nan_threshold: float in (0,1), drop any column whose fraction of
           NaN entries exceeds this threshold.
        fill_value: "median", "mean", or a float; strategy for imputing
           remaining NaNs column-wise.

    Returns:
        x_clean: numpy array of shape (N, D_kept), NaN-free.
        kept_columns: numpy array of shape (D_kept,), integer indices of the
           columns kept from the original x.
        fill_values: numpy array of shape (D_kept,), the imputation value
           used per kept column.
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
    """Convert labels from {-1, 1} to {0, 1}."""
    return np.where(y == -1, 0, 1)


def labels_to_pm1(y01):
    """Inverse of labels_to_01: convert {0,1} predictions back to {-1,1}."""
    return np.where(y01 == 0, -1, 1)


# Kept as an alias since run.py refers to this operation as
# "convert_labels_to_logistic" (labels -> {0,1} for logistic regression).
def convert_labels_to_logistic(y):
    """Convert labels from {-1, 1} to {0, 1} for logistic_regression /
    reg_logistic_regression, which expect y in {0,1}. Alias of labels_to_01."""
    return labels_to_01(y)


def standardize_columns(x, mean=None, std=None):
    """Standardize each column of x to zero mean / unit variance.

    If mean/std are None, they are computed from x (use this on x_train).
    If mean/std are given, they are applied as-is (use this on x_val/x_test
    with the TRAINING mean/std, never recomputed on val/test).
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
    """Prepend the bias column of ones: tx = [1, x_1, ..., x_D]."""
    N = x.shape[0]
    return np.c_[np.ones(N), x]


def fit_preprocessor(x, feature_names=None, missing_threshold=0.80, fill_value="median"):
    """Fit the full preprocessing pipeline (NaN-column dropping + imputation
    + standardization + bias column) on a TRAINING split, and return both
    the transformed data and the fitted parameters needed to apply the
    identical transform later to validation/test data.

    Args:
        x: numpy array of shape (N, D_raw), raw feature matrix (no Id column).
        feature_names: numpy array of shape (D_raw + 1,) including "Id" as
           the first entry (as read directly from the CSV header), or None.
        missing_threshold: float in (0,1), passed to clean_features.
        fill_value: "median", "mean", or a float, passed to clean_features.

    Returns:
        tx: numpy array of shape (N, D_kept + 1), standardized + bias column.
        retained_names: numpy array of the feature names kept (excluding
           "Id"), or None if feature_names was None.
        params: dict with "kept_columns", "fill_values", "mean", "std" --
           pass this to transform_with_preprocessor for val/test data.
    """
    x_clean, kept_columns, fill_values = clean_features(
        x, nan_threshold=missing_threshold, fill_value=fill_value
    )
    x_std, mean_x, std_x = standardize_columns(x_clean)
    tx = build_tx(x_std)

    retained_names = None
    if feature_names is not None:
        retained_names = feature_names[1:][kept_columns]

    params = {
        "kept_columns": kept_columns,
        "fill_values": fill_values,
        "mean": mean_x,
        "std": std_x,
    }
    return tx, retained_names, params


def transform_with_preprocessor(x, feature_names, params):
    """Apply a previously-fitted preprocessing pipeline (from fit_preprocessor)
    to new data (validation or test), reusing the exact column selection,
    imputation values, and standardization stats learned on the training set.

    Args:
        x: numpy array of shape (N, D_raw), raw feature matrix (no Id column).
        feature_names: numpy array of shape (D_raw + 1,) including "Id", or None.
        params: dict returned by fit_preprocessor.

    Returns:
        tx: numpy array of shape (N, D_kept + 1), standardized + bias column.
        retained_names: numpy array of the feature names kept, or None.
    """
    x_kept = x[:, params["kept_columns"]]
    nan_mask = np.isnan(x_kept)
    x_clean = np.where(nan_mask, params["fill_values"], x_kept)
    x_std, _, _ = standardize_columns(x_clean, mean=params["mean"], std=params["std"])
    tx = build_tx(x_std)

    retained_names = None
    if feature_names is not None:
        retained_names = feature_names[1:][params["kept_columns"]]

    return tx, retained_names