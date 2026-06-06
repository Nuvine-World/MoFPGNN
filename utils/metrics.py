import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def _compute_screening_single(preds, labels, thresholds):
    """
    Compute EF, NDCG, HitRate point estimates for one set of predictions.
    Used both for the observed test set and for each bootstrap replicate.

    NDCG note: labels are normalised by their maximum before the exponential
    gain (2^y - 1) to avoid numerical instability when raw activity values
    are large (e.g. max ~16 gives 2^16 ~ 63,000 without normalisation).
    Transfection efficiency is positive, so np.max(labels) is used directly.
    Normalisation is applied only when max(labels) > 10.
    """
    preds  = np.asarray(preds,  dtype=float).flatten()
    labels = np.asarray(labels, dtype=float).flatten()
    n = len(labels)

    max_label = np.max(labels)
    if max_label > 10:
        labels_ndcg = labels / max_label
    else:
        labels_ndcg = labels

    results = {}
    for p in thresholds:
        k = max(1, int(np.floor(p / 100.0 * n)))
        tag = f"{int(p)}%"

        # Top-k indices (descending)
        top_pred_idx = np.argsort(preds)[::-1][:k]
        top_true_idx = np.argsort(labels)[::-1][:k]
        hits = len(set(top_pred_idx) & set(top_true_idx))

        # EF: hits relative to random-selection expectation
        denom = k * (p / 100.0)
        results[f"EF@{tag}"] = hits / denom if denom > 0 else 0.0

        # NDCG: log2-discounted gain, normalised by ideal ranking.
        # idcg == 0 when top label is 0 (e.g. k==1 bootstrap replicate)
        ranks     = np.arange(1, k + 1)
        discounts = 1.0 / np.log2(ranks + 1)
        dcg  = float(np.sum((2.0 ** labels_ndcg[top_pred_idx] - 1.0) * discounts))
        idcg = float(np.sum((2.0 ** np.sort(labels_ndcg)[::-1][:k] - 1.0) * discounts))
        results[f"NDCG@{tag}"] = dcg / idcg if idcg > 0 else 0.0

        # HitRate: fraction of predicted top-k that are truly top-k
        results[f"HitRate@{tag}"] = hits / k if k > 0 else 0.0

    return results


def compute_screening_metrics(preds, labels, thresholds=(1, 5, 10),
                               n_bootstrap=1000, seed=42, num_floats=4):
    """
    Compute EF, NDCG, and HitRate at each top-% threshold, with 95%
    bootstrap confidence intervals (B=1000 resamples).

    For each metric two entries are written to the output dict:
        "EF@10%"      -> point estimate (float)
        "EF@10%_95CI" -> "[lower, upper]"
    """
    preds  = np.asarray(preds,  dtype=float).flatten()
    labels = np.asarray(labels, dtype=float).flatten()
    n = len(labels)

    # Point estimates on observed data
    observed    = _compute_screening_single(preds, labels, thresholds)
    metric_keys = list(observed.keys())

    # Bootstrap resampling
    rng        = np.random.default_rng(seed)
    replicates = {k: np.empty(n_bootstrap) for k in metric_keys}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)   # sample with replacement
        rep = _compute_screening_single(preds[idx], labels[idx], thresholds)
        for k in metric_keys:
            replicates[k][b] = rep[k]

    # Summarise: point estimate + 95% percentile CI
    results = {}
    for k in metric_keys:
        ci_lo = float(np.percentile(replicates[k], 2.5))
        ci_hi = float(np.percentile(replicates[k], 97.5))
        results[k] = round(float(np.mean(replicates[k])), num_floats)
        results[k + "_95CI"] = "[{}, {}]".format(
            round(ci_lo, num_floats), round(ci_hi, num_floats)
        )

    return results

def compute_regression_metrics(preds, labels, num_floats=4):
    """
    Compute standard regression metrics.

    Returns:
        dict with R2, RMSE, MAE, Pearson r
    """
    preds  = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()

    r2   = r2_score(labels, preds)
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mae  = np.mean(np.abs(preds - labels))
    corr = float(pearsonr(preds, labels).statistic)

    return {
        "R2":        round(r2,   num_floats),
        "RMSE":      round(rmse, num_floats),
        "MAE":       round(mae,  num_floats),
        "Pearson_r": round(corr, num_floats),
    }


def compute_topk_accuracy(preds, labels, k_percentiles=None):
    """
    Percentile-based ranking accuracy: fraction of compounds in the true
    top-k% that are recovered in the predicted top-k%.

    Following the protocol of Xu et al. (AGILE, Nature Communications 2024).
    """
    if k_percentiles is None:
        k_percentiles = [5, 10, 20]

    preds  = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    n = len(preds)

    results = {}
    for k in k_percentiles:
        top_n     = max(1, int(np.ceil(n * k / 100.0)))
        true_topk = set(np.argsort(labels)[-top_n:])
        pred_topk = set(np.argsort(preds)[-top_n:])
        recovery  = len(true_topk & pred_topk) / len(true_topk)
        results[f"Top-{k}%_recovery"] = round(recovery, 4)

    return results


def compute_relative_error_stats(preds, labels, percentiles=None):
    """
    Compute statistics of absolute relative error distribution.
    """
    if percentiles is None:
        percentiles = [50, 90, 95, 99]

    preds  = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()

    mask       = np.abs(labels) > 1e-8
    rel_errors = np.abs(preds[mask] - labels[mask]) / np.abs(labels[mask])

    results = {"mean_rel_error": round(float(np.mean(rel_errors)), 4)}
    for p in percentiles:
        results[f"P{p}_rel_error"] = round(float(np.percentile(rel_errors, p)), 4)

    return results


def compute_all_metrics(preds, labels, name="", num_floats=4):
    """
    Compute all metrics: regression + top-k recovery +
    screening (EF/NDCG/HitRate with 95% bootstrap CI) + relative error.
    """
    metrics = {}
    metrics.update(compute_regression_metrics(preds, labels, num_floats))
    metrics.update(compute_topk_accuracy(preds, labels))
    metrics.update(compute_screening_metrics(preds, labels, num_floats=num_floats))
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
