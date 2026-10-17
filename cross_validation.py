import numpy as np

from implementations import reg_logistic_regression, sigmoid
from evaluation import compute_metrics


def build_k_indices(y, k_fold, seed=1):
    """Build the row indices for k-fold cross-validation.

    Returns:
        numpy array of shape (k_fold, N // k_fold), each row holding the
        shuffled row indices belonging to that fold. Any remainder rows
        (N % k_fold) are simply dropped so every fold has equal size.
    """
    num_rows = len(y)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(num_rows)
    fold_size = num_rows // k_fold
    k_indices = np.array(
        [shuffled_indices[k * fold_size: (k + 1) * fold_size] for k in range(k_fold)]
    )
    return k_indices


def predict_labels_01(w, tx, threshold=0.5):
    """Predict {0,1} labels (matching the convention used internally by
    reg_logistic_regression, which trains on y in {0,1})."""
    proba = sigmoid(tx @ w)
    return (proba >= threshold).astype(float)


def cross_validate_reg_logistic(y, tx, k_fold, lambdas, gammas, max_iters, seed=1):
    """Grid search over (lambda_, gamma) for reg_logistic_regression using
    k-fold cross-validation.

    Args:
        y: numpy array of shape (N,), entries in {0, 1}.
        tx: numpy array of shape (N, D), includes bias column, standardized.
        k_fold: int, number of folds.
        lambdas: iterable of float, regularization strengths to try.
        gammas: iterable of float, step sizes to try.
        max_iters: int, fixed number of GD iterations for every run.
        seed: int, RNG seed passed to build_k_indices.

    Returns:
        results: numpy array of shape (len(lambdas), len(gammas)), the mean
           validation F1 across folds for each combination (feed this
           straight into a heatmap).
        best_lambda: float, the lambda_ achieving the best mean validation F1.
        best_gamma: float, the gamma achieving the best mean validation F1.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    lambdas = list(lambdas)
    gammas = list(gammas)
    results = np.zeros((len(lambdas), len(gammas)))

    for i, lambda_ in enumerate(lambdas):
        for j, gamma in enumerate(gammas):
            fold_scores = []
            for k in range(k_fold):
                val_idx = k_indices[k]
                tr_idx = np.hstack([k_indices[kk] for kk in range(k_fold) if kk != k])

                y_tr, tx_tr = y[tr_idx], tx[tr_idx]
                y_val, tx_val = y[val_idx], tx[val_idx]

                initial_w = np.zeros(tx.shape[1])
                w, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)

                y_val_pm1 = np.where(y_val == 0, -1, 1)
                y_pred_01 = predict_labels_01(w, tx_val)
                y_pred_pm1 = np.where(y_pred_01 == 0, -1, 1)

                score = compute_metrics(y_val_pm1, y_pred_pm1)["f1"]
                fold_scores.append(score)

            results[i, j] = np.mean(fold_scores)
            print(f"lambda={lambda_:<10} gamma={gamma:<8} mean_F1={results[i, j]:.4f}")

    best_i, best_j = np.unravel_index(np.argmax(results), results.shape)
    return results, lambdas[best_i], gammas[best_j]