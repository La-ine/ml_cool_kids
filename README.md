# ML Project 1 -- Heart Disease Prediction (CS-433, EPFL)

Binary classification of coronary heart disease (`MICHD`) from BRFSS 2015
survey data, using regularized logistic regression implemented from
scratch in NumPy. Built for the EPFL CS-433 Machine Learning course,
Project 1 (Fall 2026), submitted to the
[AIcrowd competition](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

## Project Rules

Per the course project description, this code uses **only**:

- The Python standard library
- NumPy
- Matplotlib (visualization only, e.g. CV heatmaps)

No external ML libraries (pandas, scikit-learn, PyTorch, TensorFlow, XGBoost, ...)
are used anywhere in the training/prediction pipeline, as required by the
project rules.

## Repository Structure

```
.
├── dataset/
│ ├── x_train.csv              # (328135, 322) raw training features, incl. "Id"
│ ├── y_train.csv              # (328135, 2) labels ("Id", "_MICHD" in {-1, 1})
│ └── x_test.csv               # (109379, 322) test features without labels
│
├── reports/
│ ├── class_balance.png        # generated: training class distribution
│ ├── confusion_matrix.png     # generated: validation confusion matrix
│ ├── cv_f1_grid.csv           # generated: hyperparameter tuning grid results
│ ├── feature_importance.png   # generated: top-15 standardized feature weights
│ ├── hyperparam_heatmap.png   # generated: cross-validation F1 score heatmap
│ └── run_report.txt           # generated: configuration, best hyperparameters, metrics
│
├── helpers.py                 # I/O utilities (load_csv_data, create_csv_submission)
├── implementations.py         # mandatory 6 ML baseline methods + core optimization helpers
├── preprocessing.py           # missing-value imputation, standardization, label mapping
├── cross_validation.py        # stratified k-fold splitting and hyperparameter grid search
├── evaluation.py              # classification metrics (accuracy, precision, recall, F1)
├── visualization.py           # reporting figures and diagnostic plot generators
├── run.py                     # full pipeline: loads data, runs CV, trains model, exports predictions
├── submission.csv             # generated: final test predictions formatted for AIcrowd
└── report.pdf                 # 2-page scientific conference-style project report
```

## Setup

Requires Python 3.9+ and NumPy only:

```bash
pip install numpy matplotlib
```

Download `x_train.csv`, `y_train.csv`, `x_test.csv` from the AIcrowd
competition page and place them in `dataset/` without renaming them.

## Running the Pipeline

```bash
python run.py
```

This runs the full pipeline end to end:

1. **Load** `x_train.csv` / `y_train.csv` / `x_test.csv` via `helpers.load_csv_data`.
   Any CSV row with a mismatched column count is dropped (and the matching
   `Id` is dropped from `y_train` to keep arrays aligned); a warning is
   printed naming the affected row index/indices.
2. **Stratified train/validation split** (80/20) that preserves the ~8.8%
   positive-class rate in both splits -- a plain random split risks
   skewing the already-imbalanced classes further.
3. **Preprocessing**, fit on the training split only and reused as-is on
   validation/test:
   - Drop feature columns with more than 80% missing values.
   - Impute remaining NaNs with the per-column median.
   - Standardize all features to zero mean / unit variance using
     training-set statistics.
   - Prepend the bias column.
4. **5-fold cross-validation** grid search over regularization strength
   (`lambda_`) and gradient-descent step size (`gamma`) for
   `reg_logistic_regression`, scored by mean validation F1.
5. **Final training** of `reg_logistic_regression` on the full training
   split using the best `(lambda_, gamma)` found.
6. **Validation evaluation**: accuracy, precision, recall, F1 printed and
   saved to `reports/run_report.txt`.
7. **Test prediction** and `submission.csv` generation via
   `helpers.create_csv_submission`, ready to upload to AIcrowd.

Expect the cross-validation step to take a while on the full ~328K-row,
~320-feature dataset, since this is full-batch gradient descent in pure
NumPy; reduce the `LAMBDAS` / `GAMMAS` grids or `MAX_ITERS_CV` in `run.py`
for a faster, coarser search if needed.

## The 6 Required Methods (`implementations.py`)

| Function                                                               | Description                                                           |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)`            | Linear regression, full-batch gradient descent                        |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)`           | Linear regression, SGD with batch size 1                              |
| `least_squares(y, tx)`                                                 | Least squares via normal equations                                    |
| `ridge_regression(y, tx, lambda_)`                                     | Ridge regression via normal equations                                 |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)`              | Logistic regression, gradient descent (`y` in `{0,1}`)                |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | L2-regularized logistic regression, gradient descent (`y` in `{0,1}`) |

All six return `(w, loss)`, with `w` as a 1D array of shape `(D,)` and
`loss` excluding the penalty term for the regularized variants, per the
project specification. **This model (`reg_logistic_regression`) is the
one used in `run.py`.**

## Known Data Issue

`x_train.csv` contains one malformed row (fewer columns than the header,
likely a missing field in the original BRFSS export). `helpers.load_csv_data`
detects and drops this row automatically, logging a warning; the matching
label row is dropped from `y_train.csv` to keep both arrays aligned. This
affects 1 out of 328,135 rows and has no material effect on training.

## Reproducibility Notes

- All random splits (`stratified_split` in `run.py`, `build_k_indices` in
  `cross_validation.py`) use a fixed seed (`SEED = 42`) for reproducibility.
- Preprocessing statistics (kept columns, imputation values, mean/std) are
  fit **only** on the training split and reused unchanged on validation
  and test data, to avoid data leakage.
- `submission.csv` should reproduce the score of the best AIcrowd
  submission when run against the same `dataset/` files; if scores differ
  due to later bug fixes, this is documented in `report.pdf` as required.

## Current Leaderboard Context

As of this writing, the top AIcrowd submissions score around F1 ≈ 0.455
and accuracy ≈ 0.88. Given the ~8.8% positive-class rate in the data,
there is an inherent accuracy/F1 tradeoff: a model biased toward the
majority class scores high accuracy but low F1. Hyperparameter tuning,
decision-threshold adjustment, and preprocessing choices (e.g., handling
BRFSS "don't know/refused" sentinel codes as missing data) are the main
levers explored and documented in `report.pdf`.

## Authors

ML Course Project 1, Team "ml_cool_kids" - Samarah YASSIN, Kerim TUNÇELLI, Laïne DIONGUE.

EPFL, Fall 2026.
