import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def compute_regression_metrics(preds, labels, num_floats=4):
    """
    Compute standard regression metrics.

    Returns:
        dict with R2, RMSE, MAE, Pearson r
    """
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()

    r2 = r2_score(labels, preds)
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mae = np.mean(np.abs(preds - labels))
    corr = float(pearsonr(preds, labels).statistic)

    return {
        "R2": round(r2, num_floats),
        "RMSE": round(rmse, num_floats),
        "MAE": round(mae, num_floats),
        "Pearson_r": round(corr, num_floats),
    }


def compute_topk_accuracy(preds, labels, k_percentiles=None):
    """
    Percentile-based ranking accuracy: fraction of compounds in the true
    top-k% that are recovered in the predicted top-k%.

    Following the protocol of Xu et al. (AGILE, Nature Communications 2024).

    Args:
        preds: predicted values
        labels: true values
        k_percentiles: list of percentile thresholds (default [5, 10, 20])

    Returns:
        dict mapping k -> recovery fraction
    """
    if k_percentiles is None:
        k_percentiles = [5, 10, 20]

    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    n = len(preds)

    results = {}
    for k in k_percentiles:
        top_n = max(1, int(np.ceil(n * k / 100.0)))

        # Indices of top-k% by true labels (highest mTP)
        true_topk = set(np.argsort(labels)[-top_n:])
        # Indices of top-k% by predicted values
        pred_topk = set(np.argsort(preds)[-top_n:])

        # Recovery = intersection / true_topk size
        recovery = len(true_topk & pred_topk) / len(true_topk)
        results[f"Top-{k}%_recovery"] = round(recovery, 4)

    return results


def compute_relative_error_stats(preds, labels, percentiles=None):
    """
    Compute statistics of absolute relative error distribution.

    Args:
        preds: predicted values
        labels: true values
        percentiles: percentile thresholds (default [50, 90, 95, 99])

    Returns:
        dict with mean, median, and percentile-based relative error stats
    """
    if percentiles is None:
        percentiles = [50, 90, 95, 99]

    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()

    # Avoid division by zero
    mask = np.abs(labels) > 1e-8
    rel_errors = np.abs(preds[mask] - labels[mask]) / np.abs(labels[mask])

    results = {
        "mean_rel_error": round(float(np.mean(rel_errors)), 4),
    }
    for p in percentiles:
        results[f"P{p}_rel_error"] = round(float(np.percentile(rel_errors, p)), 4)

    return results


def compute_all_metrics(preds, labels, name="", num_floats=4):
    """
    Compute all metrics (regression + ranking + relative error).

    Returns:
        dict with all metrics
    """
    metrics = {}
    metrics.update(compute_regression_metrics(preds, labels, num_floats))
    metrics.update(compute_topk_accuracy(preds, labels))
    metrics.update(compute_relative_error_stats(preds, labels))
    return metrics


def format_metrics(metrics, name=""):
    """Format metrics dict as a readable string."""
    lines = []
    if name:
        lines.append(f"\n{name}:")
    for key, val in metrics.items():
        lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def save_metrics_to_file(metrics_dict, path, title=None):
    """
    Save metrics for multiple splits to a text file.

    Args:
        metrics_dict: dict of {split_name: metrics_dict}
        path: output file path
        title: optional header string; defaults to the file's parent directory name
    """
    import os
    if title is None:
        title = os.path.basename(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(f"{title} -- Evaluation Metrics\n")
        f.write("=" * 60 + "\n")
        for name, metrics in metrics_dict.items():
            f.write(format_metrics(metrics, name))
            f.write("\n")
    print(f"Metrics saved to {path}")
