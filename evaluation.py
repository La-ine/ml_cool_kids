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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def format_results_table(results_dict):
    """Format a dict of metric_name -> value (or per-fold list) into an
    aligned plain-text table suitable for printing to terminal or writing
    to a .txt report.

    Args:
        results_dict: dict mapping str metric name to either a float, int,
           str, or a list/array of per-fold floats.

    Returns:
        str, a formatted multi-line table.
    """
    lines = []
    key_width = max(len(k) for k in results_dict) + 2

    for key, value in results_dict.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value, dtype=float)
            val_str = (
                f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
                f"per_fold={np.round(arr, 4).tolist()}"
            )
        elif isinstance(value, float):
            val_str = f"{value:.4f}"
        else:
            val_str = str(value)
        lines.append(f"{key:<{key_width}}: {val_str}")

    width = max(len(line) for line in lines) + 4
    sep = "-" * width
    return sep + "\n" + "\n".join(lines) + "\n" + sep


def save_run_report(path, config, metrics, best_hyperparams):
    """Write a full run report (config + metrics + chosen hyperparameters)
    to a text file for record-keeping across experiments.

    Args:
        path: str, output file path (e.g. "reports/run_2026-10-14.txt").
        config: dict, run configuration (model name, max_iters, k_fold, etc.).
        metrics: dict, output of compute_metrics (or cross-val summary).
        best_hyperparams: dict, e.g. {"lambda_": ..., "gamma": ...}.

    Returns:
        None. Writes the report to disk at `path`.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    sections = []
    sections.append("=" * 60)
    sections.append("RUN CONFIGURATION")
    sections.append("=" * 60)
    sections.append(format_results_table(config))
    sections.append("")
    sections.append("=" * 60)
    sections.append("BEST HYPERPARAMETERS")
    sections.append("=" * 60)
    sections.append(format_results_table(best_hyperparams))
    sections.append("")
    sections.append("=" * 60)
    sections.append("METRICS")
    sections.append("=" * 60)
    sections.append(format_results_table(metrics))
    sections.append("")

    with open(path, "w") as f:
        f.write("\n".join(sections))