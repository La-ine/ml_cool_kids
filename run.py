import csv
import numpy as np

from helpers import load_csv_data
from preprocessing import (
    fit_preprocessor,
    transform_with_preprocessor,
    convert_labels_to_logistic,
)
from implementations import logistic_regression


def main(): 

    # Load data
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        "dataset/"
    )

    with open("dataset/x_train.csv", "r") as f:
        reader = csv.reader(f)
        feature_names = np.array(next(reader))

    # Split train / validation ##ATTENTION: il faudrait changer le split car il y a uniquement 8.8% des labels qui sont positifs. Donc il faudrait faire un stratified split pour que la proportion de labels positifs soit la même dans le train et le validation.
    np.random.seed(42)

    indices = np.random.permutation(len(y_train))
    split = int(0.8 * len(y_train))

    train_indices = indices[:split]
    val_indices = indices[split:]

    x_tr = x_train[train_indices]
    y_tr = y_train[train_indices]

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]

    # Fit preprocessing on training split only
    x_tr_processed, retained_names, params = fit_preprocessor(
        x_tr,
        feature_names,
        missing_threshold=0.80,
    )

    # Apply the same preprocessing to validation
    x_val_processed, _ = transform_with_preprocessor(
        x_val,
        feature_names,
        params,
    )

    # Convert labels for logistic regression
    y_tr_logistic = convert_labels_to_logistic(y_tr)
    y_val_logistic = convert_labels_to_logistic(y_val)

    print("Processed training shape:", x_tr_processed.shape)
    print("Processed validation shape:", x_val_processed.shape)


if __name__ == "__main__":
    main()
