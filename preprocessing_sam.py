import numpy as np


def mode_ignore_nan(column):
    """Return the most frequent non-missing value in a 1D array.

    NaN values are ignored when computing the mode. This helper is mainly
    used to impute binary and categorical features.

    Args:
        column: numpy array of shape (N,), possibly containing NaN values.

    Returns:
        mode: scalar, the most frequent non-NaN value in the column.

    Raises:
        ValueError: if the column contains only NaN values.
    """
    valid = column[~np.isnan(column)]

    if len(valid) == 0:
        raise ValueError("Cannot compute mode of a fully-missing column.")

    values, counts = np.unique(valid, return_counts=True)
    return values[np.argmax(counts)]


def infer_feature_types(x, names, categorical_threshold=20):
    """Infer a simple feature type for each column.

    This is a baseline heuristic based only on the number of unique
    non-missing values:
        - <= 2 unique values: binary
        - <= categorical_threshold unique values: categorical
        - otherwise: continuous

    This does not perfectly recover the semantic type of every BRFSS
    variable. It is intended as a practical first baseline and can later
    be overridden manually for known ordinal or specially encoded features.

    Args:
        x: numpy array of shape (N, D), feature matrix.
        names: numpy array of shape (D,), feature names.
        categorical_threshold: integer, maximum number of unique values
            for a feature to be classified as categorical.

    Returns:
        feature_types: dictionary mapping each feature name to one of
            {"binary", "categorical", "continuous"}.
    """
    feature_types = {}

    for j, name in enumerate(names):
        column = x[:, j]
        valid = column[~np.isnan(column)]
        n_unique = len(np.unique(valid))

        if n_unique <= 2:
            feature_types[name] = "binary"
        elif n_unique <= categorical_threshold:
            feature_types[name] = "categorical"
        else:
            feature_types[name] = "continuous"

    return feature_types


def compute_missing_rates(x, names):
    """Compute the fraction of missing values for every feature.

    Args:
        x: numpy array of shape (N, D), feature matrix containing NaNs
            for missing observations.
        names: numpy array of shape (D,), feature names.

    Returns:
        missing_rates: dictionary mapping each feature name to its missing
            value rate, expressed as a number between 0 and 1.
    """
    return {
        name: np.mean(np.isnan(x[:, j]))
        for j, name in enumerate(names)
    }


def fit_missing_filter(x, names, missing_threshold=0.80):
    """Determine which features to keep based on training-set missingness.

    A feature is removed when its missing-value rate is strictly greater
    than missing_threshold.

    This function should be called on the TRAINING data only. The resulting
    keep_mask must then be reused unchanged on validation and test data.

    Args:
        x: numpy array of shape (N, D), training feature matrix.
        names: numpy array of shape (D,), feature names.
        missing_threshold: float between 0 and 1. Features with a missing
            rate greater than this value are removed.

    Returns:
        keep_mask: boolean numpy array of shape (D,), True for retained
            columns and False for removed columns.
        retained_names: numpy array containing names of retained features.
        removed_names: numpy array containing names of removed features.
        missing_rates: dictionary mapping each feature name to its training
            missing-value rate.
    """
    missing_rates = compute_missing_rates(x, names)

    keep_mask = np.array([
        missing_rates[name] <= missing_threshold
        for name in names
    ])

    retained_names = names[keep_mask]
    removed_names = names[~keep_mask]

    return keep_mask, retained_names, removed_names, missing_rates


def apply_column_mask(x, names, keep_mask):
    """Apply a previously learned feature-selection mask.

    This is useful for applying the exact same column selection learned from
    the training data to validation or test data.

    Args:
        x: numpy array of shape (N, D), feature matrix.
        names: numpy array of shape (D,), feature names.
        keep_mask: boolean numpy array of shape (D,), usually learned from
            fit_missing_filter on the training set.

    Returns:
        x_filtered: numpy array of shape (N, D_kept), containing only the
            retained columns.
        retained_names: numpy array of shape (D_kept,), names of retained
            features.
    """
    return x[:, keep_mask], names[keep_mask]


def fit_imputer(x, names, feature_types):
    """Learn one imputation value per feature from training data.

    Continuous features are imputed with their median.
    Binary and categorical features are imputed with their mode.

    This function should be called on TRAINING data only. The learned
    imputation values must be reused unchanged on validation and test data.

    Args:
        x: numpy array of shape (N, D), training feature matrix.
        names: numpy array of shape (D,), feature names.
        feature_types: dictionary mapping each feature name to its inferred
            type, e.g. "binary", "categorical", or "continuous".

    Returns:
        imputation_values: dictionary mapping feature names to the scalar
            value that should replace NaNs in that feature. Features with no
            missing values are not included in the dictionary.
    """
    imputation_values = {}

    for j, name in enumerate(names):
        column = x[:, j]

        if not np.any(np.isnan(column)):
            continue

        if feature_types[name] == "continuous":
            value = np.nanmedian(column)
        else:
            value = mode_ignore_nan(column)

        imputation_values[name] = value

    return imputation_values


def apply_imputer(x, names, imputation_values):
    """Replace missing values using previously learned imputation values.

    The imputation_values should normally come from fit_imputer applied to
    the training set. They should not be recomputed on validation or test
    data, in order to avoid data leakage.

    Args:
        x: numpy array of shape (N, D), feature matrix containing NaNs.
        names: numpy array of shape (D,), feature names.
        imputation_values: dictionary mapping feature names to the scalar
            values used for imputation.

    Returns:
        x_imputed: numpy array of shape (N, D), with available NaN values
            replaced by their corresponding learned imputation value.
    """
    x_imputed = x.copy()

    for j, name in enumerate(names):
        if name not in imputation_values:
            continue

        missing = np.isnan(x_imputed[:, j])
        x_imputed[missing, j] = imputation_values[name]

    return x_imputed


def fit_standardizer(x, names, feature_types):
    """Learn mean and standard deviation for continuous training features.

    Only features classified as continuous are standardized. Binary and
    categorical features are left unchanged.

    This function should be called after imputation, so that no NaN values
    remain in the continuous columns. It should be fitted on TRAINING data
    only and the learned statistics reused on validation/test data.

    Args:
        x: numpy array of shape (N, D), imputed training feature matrix.
        names: numpy array of shape (D,), feature names.
        feature_types: dictionary mapping feature names to feature types.

    Returns:
        means: dictionary mapping continuous feature names to their training
            mean.
        stds: dictionary mapping continuous feature names to their training
            standard deviation. A standard deviation of 0 is replaced by 1
            to avoid division by zero.
    """
    means = {}
    stds = {}

    for j, name in enumerate(names):
        if feature_types[name] != "continuous":
            continue

        mean = np.mean(x[:, j])
        std = np.std(x[:, j])

        if std == 0:
            std = 1.0

        means[name] = mean
        stds[name] = std

    return means, stds


def apply_standardizer(x, names, means, stds):
    """Standardize continuous features using learned training statistics.

    For each continuous feature, this applies:
        (x - training_mean) / training_std

    The means and standard deviations should come from fit_standardizer on
    the training set. They must not be recomputed on validation or test data.

    Args:
        x: numpy array of shape (N, D), imputed feature matrix.
        names: numpy array of shape (D,), feature names.
        means: dictionary mapping continuous feature names to training means.
        stds: dictionary mapping continuous feature names to training standard
            deviations.

    Returns:
        x_scaled: numpy array of shape (N, D), with continuous features
            standardized and other feature types unchanged.
    """
    x_scaled = x.copy()

    for j, name in enumerate(names):
        if name not in means:
            continue

        x_scaled[:, j] = (
            x_scaled[:, j] - means[name]
        ) / stds[name]

    return x_scaled


def fit_preprocessor(x_train, feature_names, missing_threshold=0.80, categorical_threshold=20):
    """Fit the complete baseline preprocessing pipeline on training data.

    The pipeline performs the following steps:
        1. Remove columns with too much missing data.
        2. Infer rough feature types.
        3. Learn imputation values.
        4. Impute missing values.
        5. Learn means/stds for continuous features.
        6. Standardize continuous features.

    This function must be fitted on the TRAINING split only. The returned
    parameters should then be passed to transform_with_preprocessor for the
    validation and test sets.

    Args:
        x_train: numpy array of shape (N, D), raw training feature matrix.
        feature_names: numpy array of shape (D,), original feature names.
        missing_threshold: float between 0 and 1. Features with a larger
            missing rate are removed.
        categorical_threshold: integer, maximum number of unique values for
            a feature to be automatically classified as categorical.

    Returns:
        x_processed: numpy array of shape (N, D_kept), processed training
            features.
        retained_names: numpy array of shape (D_kept,), names of retained
            features.
        params: dictionary containing all preprocessing information learned
            from the training data, including:
                - keep_mask
                - removed_feature_names
                - missing_rates
                - feature_types
                - imputation_values
                - means
                - stds
                - missing_threshold
                - categorical_threshold
    """
    keep_mask, retained_names, removed_names, missing_rates = fit_missing_filter(
        x_train,
        feature_names,
        missing_threshold=missing_threshold,
    )

    x_filtered = x_train[:, keep_mask]

    feature_types = infer_feature_types(
        x_filtered,
        retained_names,
        categorical_threshold=categorical_threshold,
    )

    imputation_values = fit_imputer(
        x_filtered,
        retained_names,
        feature_types,
    )

    x_imputed = apply_imputer(
        x_filtered,
        retained_names,
        imputation_values,
    )

    means, stds = fit_standardizer(
        x_imputed,
        retained_names,
        feature_types,
    )

    x_processed = apply_standardizer(
        x_imputed,
        retained_names,
        means,
        stds,
    )

    params = {
        "keep_mask": keep_mask,
        "removed_feature_names": removed_names,
        "missing_rates": missing_rates,
        "feature_types": feature_types,
        "imputation_values": imputation_values,
        "means": means,
        "stds": stds,
        "missing_threshold": missing_threshold,
        "categorical_threshold": categorical_threshold,
    }

    return x_processed, retained_names, params


def transform_with_preprocessor(x, feature_names, params):
    """Apply a fitted preprocessing pipeline to validation or test data.

    This function reuses all preprocessing decisions learned from the training
    data. It does not recompute missingness thresholds, imputation values,
    feature types, means, or standard deviations.

    Args:
        x: numpy array of shape (N, D), raw validation or test feature matrix.
        feature_names: numpy array of shape (D,), original feature names in the
            same order used when fitting the preprocessor.
        params: dictionary returned by fit_preprocessor.

    Returns:
        x_processed: numpy array of shape (N, D_kept), transformed feature
            matrix.
        retained_names: numpy array of shape (D_kept,), retained feature names.
    """
    keep_mask = params["keep_mask"]

    x_filtered = x[:, keep_mask]
    retained_names = feature_names[keep_mask]

    x_imputed = apply_imputer(
        x_filtered,
        retained_names,
        params["imputation_values"],
    )

    x_processed = apply_standardizer(
        x_imputed,
        retained_names,
        params["means"],
        params["stds"],
    )

    return x_processed, retained_names


def convert_labels_to_logistic(y):
    """Convert project labels {-1, +1} into logistic labels {0, 1}.

    The Project 1 logistic-regression implementations expect binary target
    values encoded as 0 and 1.

    Args:
        y: numpy array of shape (N,), containing labels -1 and +1.

    Returns:
        y_logistic: numpy array of shape (N,), where -1 is mapped to 0 and
            +1 is mapped to 1. The returned dtype is float.
    """
    return (y == 1).astype(float)
