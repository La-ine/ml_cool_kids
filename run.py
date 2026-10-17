"""
run.py -- Full training + evaluation + visualization + submission pipeline
for the EPFL CS-433 Project 1 heart-disease (MICHD) classification task.

Usage (from the project root, with dataset/x_train.csv, dataset/y_train.csv,
dataset/x_test.csv all present):

    python run.py

Produces:
    - submission.csv                      (ready to upload to aicrowd.com)
    - reports/run_report.txt              (config, best hyperparameters, metrics)
    - reports/cv_f1_grid.csv               (F1 grid, for reference)
    - reports/hyperparam_heatmap.png       (F1 across lambda x gamma grid)
    - reports/class_balance.png            (class distribution)
    - reports/confusion_matrix.png         (validation confusion matrix)
    - reports/feature_importance.png       (top-k logistic regression weights)
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from helpers import load_csv_data, create_csv_submission
from preprocessing import (
    fit_preprocessor,
    transform_with_preprocessor,
    convert_labels_to_logistic,
    labels_to_pm1,
)
from implementations import reg_logistic_regression, sigmoid
from cross_validation import cross_validate_reg_logistic
from evaluation import compute_metrics, format_results_table, save_run_report
from visualization import (
    plot_hyperparam_heatmap,
    plot_class_balance,
    plot_confusion_matrix,
    plot_feature_importance
)

DATA_PATH = "dataset"
MISSING_THRESHOLD = 0.80
K_FOLD = 5
MAX_ITERS_CV = 200
MAX_ITERS_FINAL = 2000
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
GAMMAS = [0.01, 0.05, 0.1, 0.3]
SEED = 42


def stratified_split(y, ratio=0.8, seed=SEED):
    """Split indices into train/val while preserving the class balance of y
    (critical here since only ~8.8% of labels are positive)."""
    np.random.seed(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_pos_tr = int(len(pos_idx) * ratio)
    n_neg_tr = int(len(neg_idx) * ratio)

    train_idx = np.concatenate([pos_idx[:n_pos_tr], neg_idx[:n_neg_tr]])
    val_idx = np.concatenate([pos_idx[n_pos_tr:], neg_idx[n_neg_tr:]])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    return train_idx, val_idx


def predict_labels_pm1(w, tx, threshold=0.5):
    """Predict {-1, 1} labels from a weight vector trained on {0,1} targets."""
    proba = sigmoid(tx @ w)
    y01 = (proba >= threshold).astype(float)
    return labels_to_pm1(y01)


def main():
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    x_train_raw, x_test_raw, y_train, train_ids, test_ids = load_csv_data(
        DATA_PATH, sub_sample=False
    )
    with open(f"{DATA_PATH}/x_train.csv", "r") as f:
        reader = csv.reader(f)
        feature_names = np.array(next(reader))

    assert len(y_train) == len(x_train_raw) == len(train_ids), \
        f"Length mismatch: x={len(x_train_raw)}, y={len(y_train)}, ids={len(train_ids)}"
    assert len(np.unique(train_ids)) == len(train_ids), "Duplicate IDs in train_ids"
    assert np.all(np.diff(train_ids) > 0), "train_ids not monotonically increasing"

    print("x_train_raw:", x_train_raw.shape, "| x_test_raw:", x_test_raw.shape)
    print("y_train unique:", np.unique(y_train),
          "| positive rate:", np.mean(y_train == 1))

    print("\n" + "=" * 60)
    print("STEP 2: Stratified train/validation split")
    print("=" * 60)
    train_idx, val_idx = stratified_split(y_train, ratio=0.8, seed=SEED)
    x_tr_raw, y_tr = x_train_raw[train_idx], y_train[train_idx]
    x_val_raw, y_val = x_train_raw[val_idx], y_train[val_idx]
    print("Train:", x_tr_raw.shape, "| Val:", x_val_raw.shape)
    print("Train positive rate:", np.mean(y_tr == 1),
          "| Val positive rate:", np.mean(y_val == 1))

    print("\n" + "=" * 60)
    print("STEP 3: Fit preprocessing on training split, apply to val/test")
    print("=" * 60)
    tx_tr, retained_names, params = fit_preprocessor(
        x_tr_raw, feature_names, missing_threshold=MISSING_THRESHOLD
    )
    tx_val, _ = transform_with_preprocessor(x_val_raw, feature_names, params)
    tx_test, _ = transform_with_preprocessor(x_test_raw, feature_names, params)

    print("tx_tr:", tx_tr.shape, "| tx_val:", tx_val.shape, "| tx_test:", tx_test.shape)
    print(f"Kept {len(params['kept_columns'])} / {x_train_raw.shape[1]} raw features "
          f"(threshold={MISSING_THRESHOLD})")

    y_tr_01 = convert_labels_to_logistic(y_tr)
    print("Verified training samples:", tx_tr.shape[0])
    print("Verified validation samples:", tx_val.shape[0]) 

    print("\n" + "=" * 60)
    print(f"STEP 4: {K_FOLD}-fold CV grid search over lambda x gamma")
    print("=" * 60)
    cv_results, best_lambda, best_gamma = cross_validate_reg_logistic(
        y_tr_01, tx_tr, k_fold=K_FOLD, lambdas=LAMBDAS, gammas=GAMMAS,
        max_iters=MAX_ITERS_CV, seed=SEED,
    )
    print(f"\nBest hyperparameters: lambda_={best_lambda}, gamma={best_gamma}")
    np.savetxt("reports/cv_f1_grid.csv", cv_results, delimiter=",")

    print("\n" + "=" * 60)
    print("STEP 5: Train final model on full training split")
    print("=" * 60)
    initial_w = np.zeros(tx_tr.shape[1])
    w, train_loss = reg_logistic_regression(
        y_tr_01, tx_tr, best_lambda, initial_w, MAX_ITERS_FINAL, best_gamma
    )
    print("Final training loss:", train_loss)

    print("\n" + "=" * 60)
    print("STEP 6: Evaluate on held-out validation split")
    print("=" * 60)
    y_val_pred = predict_labels_pm1(w, tx_val)
    val_metrics = compute_metrics(y_val, y_val_pred)
    print(format_results_table(val_metrics))

    print("\n" + "=" * 60)
    print("STEP 7: Predict on test set and write submission")
    print("=" * 60)
    y_test_pred = predict_labels_pm1(w, tx_test)
    create_csv_submission(test_ids, y_test_pred, "submission.csv")
    print("Wrote submission.csv with", len(test_ids), "predictions")
    print("Predicted positive rate on test:", np.mean(y_test_pred == 1))

    print("\n" + "=" * 60)
    print("STEP 8: Generate report visualizations")
    print("=" * 60)
    os.makedirs("reports", exist_ok=True)

    # 8a. Hyperparameter grid heatmap (uses the CV results from Step 4)
    fig = plot_hyperparam_heatmap(cv_results, LAMBDAS, GAMMAS, metric_name="F1")
    plt.close(fig)
    print("Saved reports/hyperparam_heatmap.png")

    # 8b. Class balance (motivates stratified split / threshold tuning)
    fig = plot_class_balance(y_train)
    plt.close(fig)
    print("Saved reports/class_balance.png")

    # 8c. Confusion matrix on the validation split
    fig = plot_confusion_matrix(y_val, y_val_pred)
    plt.close(fig)
    print("Saved reports/confusion_matrix.png")

    # 8d. Feature importance (top-15 |w|, valid since features are standardized)
    if retained_names is not None:
        fig = plot_feature_importance(w, retained_names, top_k=15)
        plt.close(fig)
        print("Saved reports/feature_importance.png")

    print("\n" + "=" * 60)
    print("STEP 9: Save run report")
    print("=" * 60)
    config = {
        "model": "reg_logistic_regression",
        "missing_threshold": MISSING_THRESHOLD,
        "k_fold": K_FOLD,
        "max_iters_cv": MAX_ITERS_CV,
        "max_iters_final": MAX_ITERS_FINAL,
        "n_features_kept": len(params["kept_columns"]),
        "train_size": len(y_tr),
        "val_size": len(y_val),
    }
    best_hyperparams = {"lambda_": best_lambda, "gamma": best_gamma}
    save_run_report("reports/run_report.txt", config, val_metrics, best_hyperparams)
    print("Report written to reports/run_report.txt")


if __name__ == "__main__":
    main()
