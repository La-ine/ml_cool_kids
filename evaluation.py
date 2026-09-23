import numpy as np


def compute_metrics(y_true, y_pred):
    """Compute classification metrics for labels in {-1, 1}.

    Args:
        y_true: numpy array of shape (N,), ground-truth labels in {-1, 1}.
        y_pred: numpy array of shape (N,), predicted labels in {-1, 1}.

    Returns:
        dict with keys "accuracy", "precision", "recall", "f1", each a float,
        computed with the positive class defined as y == 1 (heart disease).
    """
    raise NotImplementedError


def format_results_table(results_dict):
    """Format a dict of metric_name -> value (or per-fold list) into an
    aligned plain-text table suitable for printing to terminal or writing
    to a .txt report.

    Args:
        results_dict: dict mapping str metric name to either a float or a
           list/array of per-fold floats.

    Returns:
        str, a formatted multi-line table (e.g. built with str.format /
        f-strings and column alignment, or via tabulate if available).
    """
    raise NotImplementedError


def save_run_report(path, config, metrics, best_hyperparams):
    """Write a full run report (config + metrics + chosen hyperparameters)
    to a text file for record-keeping across experiments.

    Args:
        path: str, output file path (e.g. "reports/run_2026-09-22.txt").
        config: dict, run configuration (model name, max_iters, k_fold, etc.).
        metrics: dict, output of compute_metrics (or cross-val summary).
        best_hyperparams: dict, e.g. {"lambda_": ..., "gamma": ...}.

    Returns:
        None. Writes the report to disk at `path`.
    """
    raise NotImplementedError