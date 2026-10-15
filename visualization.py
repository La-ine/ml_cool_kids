"""
visualization.py -- Plotting functions for the Project 1 report.

All functions use matplotlib only (allowed per the course rules for
visualization purposes), save a .png to disk, and return the matplotlib
Figure object in case you want to tweak it further (e.g. in a notebook)
before saving. None of these functions perform any ML computation --
they take already-computed arrays/dicts as input, matching the outputs
of cross_validation.py and evaluation.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_hyperparam_heatmap(results, lambdas, gammas, metric_name="F1",
                             save_path="reports/hyperparam_heatmap.png"):
    """Plot a 2D heatmap of cross-validation scores over a (lambda, gamma)
    grid, as returned by cross_validation.cross_validate_reg_logistic.

    Args:
        results: numpy array of shape (len(lambdas), len(gammas)), the
           mean validation score for each hyperparameter combination.
        lambdas: list of float, regularization strengths (rows).
        gammas: list of float, step sizes (columns).
        metric_name: str, name of the metric in results (for axis label).
        save_path: str, where to save the figure.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(1.4 * len(gammas) + 2, 1.0 * len(lambdas) + 2))
    im = ax.imshow(results, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels(gammas)
    ax.set_yticks(range(len(lambdas)))
    ax.set_yticklabels(lambdas)
    ax.set_xlabel("gamma (step size)")
    ax.set_ylabel("lambda_ (regularization strength)")
    ax.set_title(f"{metric_name} score across hyperparameter grid")

    vmin, vmax = results.min(), results.max()
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            color = "white" if results[i, j] < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{results[i, j]:.3f}", ha="center", va="center",
                     color=color, fontsize=9)

    best_i, best_j = np.unravel_index(np.argmax(results), results.shape)
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                  fill=False, edgecolor="red", linewidth=2.5))

    plt.colorbar(im, ax=ax, label=f"Mean validation {metric_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return fig


def plot_class_balance(y, save_path="reports/class_balance.png"):
    """Bar chart of class counts/proportions for labels in {-1, 1}.
    Useful for motivating the imbalance-aware choices in the report
    (stratified split, threshold tuning, etc.).

    Args:
        y: numpy array of shape (N,), entries in {-1, 1}.
        save_path: str, where to save the figure.

    Returns:
        matplotlib.figure.Figure
    """
    n_neg = int(np.sum(y == -1))
    n_pos = int(np.sum(y == 1))
    total = n_neg + n_pos

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(["No disease (-1)", "Disease (1)"], [n_neg, n_pos],
                     color=["#4C72B0", "#C44E52"])
    for b, c in zip(bars, [n_neg, n_pos]):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({c/total*100:.1f}%)",
                  ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Class distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path="reports/confusion_matrix.png"):
    """Plot a 2x2 confusion matrix for labels in {-1, 1}.

    Args:
        y_true: numpy array of shape (N,), ground-truth labels in {-1, 1}.
        y_pred: numpy array of shape (N,), predicted labels in {-1, 1}.
        save_path: str, where to save the figure.

    Returns:
        matplotlib.figure.Figure
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    cm = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center",
                      fontsize=11, color=color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: -1", "Pred: 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True: -1", "True: 1"])
    ax.set_title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return fig


def plot_feature_importance(w, feature_names, top_k=15,
                             save_path="reports/feature_importance.png"):
    """Horizontal bar chart of the top-k largest-magnitude logistic
    regression weights, as a simple feature-importance proxy (valid only
    because features were standardized before training, so weight
    magnitudes are directly comparable).

    Args:
        w: numpy array of shape (D+1,), learned weight vector INCLUDING
           the bias term at index 0.
        feature_names: numpy array of shape (D,), names of the retained
           features in the same order used to build tx (i.e. excluding
           "Id" and excluding the bias column).
        top_k: int, number of top features to display.
        save_path: str, where to save the figure.

    Returns:
        matplotlib.figure.Figure
    """
    w_features = w[1:]  # drop bias term
    order = np.argsort(-np.abs(w_features))[:top_k]

    fig, ax = plt.subplots(figsize=(7, 0.35 * top_k + 1.5))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in w_features[order][::-1]]
    ax.barh(feature_names[order][::-1], w_features[order][::-1], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Learned weight (standardized features)")
    ax.set_title(f"Top {top_k} features by |weight|")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return fig


def plot_ablation_comparison(labels, scores, metric_name="F1",
                               save_path="reports/ablation_comparison.png"):
    """Bar chart comparing validation scores across an ablation study
    (e.g. baseline -> + NaN handling -> + standardization -> + reg tuning
    -> + threshold tuning), directly supporting the "ablation study" and
    "baselines" grading criteria in the report.

    Args:
        labels: list of str, one name per experiment/configuration.
        scores: list of float, the corresponding validation score
           (e.g. F1) for each configuration, same order as labels.
        metric_name: str, for the y-axis label.
        save_path: str, where to save the figure.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 4.5))
    bars = ax.bar(labels, scores, color="#4C72B0")
    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"Validation {metric_name}")
    ax.set_title("Ablation study: impact of each modification")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return fig
