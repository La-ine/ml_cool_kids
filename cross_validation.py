import numpy as np


def split_data(y, tx, ratio=0.8, seed=1):
    """Randomly split (y, tx) into a training and validation subset.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N, D).
        ratio: float in (0,1), fraction of rows assigned to training.
        seed: int, RNG seed for reproducibility.

    Returns:
        y_tr, tx_tr, y_val, tx_val: split arrays with matching row counts
        (round(N*ratio) in train, remainder in val).
    """
    num_tr_samples = int(len(tx) * ratio)
    
    # Shuffle indices
    indices = np.random.permutation(len(tx))
    train_indices = indices[:num_tr_samples]
    val_indices = indices[num_tr_samples:]
    
    # Create validation set
    tx_val = tx[val_indices]
    y_val = y[val_indices]
    
    # Update training set
    tx_tr = tx[train_indices]
    y_tr = y[train_indices]
    return y_tr, tx_tr, y_val, tx_val


def build_k_indices(y, k_fold, seed=1):
    """Build the row indices for k-fold cross-validation.

    Args:
        y: numpy array of shape (N,).
        k_fold: int, number of folds.
        seed: int, RNG seed.

    Returns:
        numpy array of shape (k_fold, N // k_fold), each row holding the
        shuffled row indices belonging to that fold.
    """
    raise NotImplementedError


def cross_validate_reg_logistic(y, tx, k_fold, lambdas, gammas, max_iters, seed=1):
    """Grid search over (lambda_, gamma) for reg_logistic_regression using
    k-fold cross-validation.

    Args:
        y: numpy array of shape (N,), entries in {0, 1}.
        tx: numpy array of shape (N, D), includes bias column, standardized.
        k_fold: int, number of folds.
        lambdas: iterable of float, regularization strengths to try.
        gammas: iterable of float, step sizes to try.
        max_iters: int, fixed number of GD iterations for every run
           (kept fixed during the grid search; large enough to converge).
        seed: int, RNG seed passed to build_k_indices.

    Returns:
        results: numpy array of shape (len(lambdas), len(gammas)), the mean
           validation F1 (or chosen metric) across folds for each combination
           -- this is exactly the grid you would feed into a heatmap plot.
        best_lambda: float, the lambda_ achieving the best mean validation score.
        best_gamma: float, the gamma achieving the best mean validation score.
    """
    raise NotImplementedError