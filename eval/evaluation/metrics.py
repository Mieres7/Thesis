"""Classification, fairness, and final-run AUC diagnostics."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _canonical_k(y_proba: np.ndarray, labels: list | None) -> int:
    return len(labels) if labels is not None else int(y_proba.shape[1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                    labels: list | None = None) -> dict:
    """Standard metrics, with AUC evaluated in the canonical label space."""
    n_classes = _canonical_k(y_proba, labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if n_classes == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            metrics["auc_ovr_macro"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=labels
            ))
    except (ValueError, IndexError) as exc:
        metrics["auc" if n_classes == 2 else "auc_ovr_macro"] = None
        metrics["auc_error"] = str(exc)

    cm_labels = labels if labels is not None else sorted(
        np.unique(np.concatenate([y_true, y_pred])).tolist()
    )
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    sensitivity = {}
    for i, cls in enumerate(cm_labels):
        support = int(cm[i, :].sum())
        sensitivity[f"class_{cls}"] = float(cm[i, i] / support) if support else 0.0
    metrics["per_class_sensitivity"] = sensitivity
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             labels: list | None = None) -> dict:
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for i, cls in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(cls)] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(cm[i, :].sum()),
        }
    return {"confusion_matrix": cm.tolist(), "per_class": per_class,
            "labels": [str(label) for label in labels]}


def compute_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                               labels: list | None = None) -> dict:
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = [
        (str(true_cls), str(pred_cls), int(cm[i, j]))
        for i, true_cls in enumerate(labels)
        for j, pred_cls in enumerate(labels)
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda value: value[2], reverse=True)
    total = int(cm.sum())
    errors = int(total - np.trace(cm))
    return {"total_errors": errors, "total_samples": total,
            "error_rate": errors / total if total else 0.0,
            "confused_pairs": pairs, "most_confused": pairs[0] if pairs else None}


def compute_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    if len(labels) < 2:
        return {"tpr": None, "fpr": None}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tpr, fpr = [], []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        tpr.append(tp / (tp + fn) if tp + fn else 0.0)
        fpr.append(fp / (fp + tn) if fp + tn else 0.0)
    return {"tpr": float(np.mean(tpr)), "fpr": float(np.mean(fpr))}


def _max_gap(subgroup_metrics: dict[str, float], metric_name: str) -> dict:
    valid = {key: value for key, value in subgroup_metrics.items() if value is not None}
    if len(valid) < 2:
        return {f"delta_{metric_name}": None, "best_group": None, "worst_group": None,
                f"best_{metric_name}": None, f"worst_{metric_name}": None}
    best = max(valid, key=valid.get)
    worst = min(valid, key=valid.get)
    return {f"delta_{metric_name}": float(valid[best] - valid[worst]),
            "best_group": best, "worst_group": worst,
            f"best_{metric_name}": float(valid[best]), f"worst_{metric_name}": float(valid[worst])}


def compute_delta_auc(subgroup_metrics: dict[str, float]) -> dict:
    return _max_gap(subgroup_metrics, "auc")


def compute_delta_ba(subgroup_metrics: dict[str, float]) -> dict:
    return _max_gap(subgroup_metrics, "ba")


def compute_delta_f1(subgroup_metrics: dict[str, float]) -> dict:
    return _max_gap(subgroup_metrics, "f1")


def compute_average_gap(subgroup_metrics: dict[str, float], global_metric: float) -> dict:
    valid = {key: value for key, value in subgroup_metrics.items() if value is not None}
    if len(valid) < 2 or global_metric is None:
        return {"avg_gap": None, "per_group": None}
    per_group = {key: abs(value - global_metric) for key, value in valid.items()}
    return {"avg_gap": float(np.mean(list(per_group.values()))), "per_group": per_group}


def compute_equalized_odds_gap(subgroup_tpr: dict[str, float],
                                subgroup_fpr: dict[str, float]) -> dict:
    valid_tpr = {key: value for key, value in subgroup_tpr.items() if value is not None}
    valid_fpr = {key: value for key, value in subgroup_fpr.items() if value is not None}
    result = {"tpr_gap": None, "fpr_gap": None, "tpr_best_group": None,
              "tpr_worst_group": None, "fpr_best_group": None, "fpr_worst_group": None}
    if len(valid_tpr) >= 2:
        best, worst = max(valid_tpr, key=valid_tpr.get), min(valid_tpr, key=valid_tpr.get)
        result.update(tpr_gap=float(valid_tpr[best] - valid_tpr[worst]),
                      tpr_best_group=best, tpr_worst_group=worst)
    if len(valid_fpr) >= 2:
        best, worst = min(valid_fpr, key=valid_fpr.get), max(valid_fpr, key=valid_fpr.get)
        result.update(fpr_gap=float(valid_fpr[worst] - valid_fpr[best]),
                      fpr_best_group=best, fpr_worst_group=worst)
    return result


def compute_auc_es(global_auc: float, delta_auc: float, lambda_param: float = 1.0) -> float | None:
    if global_auc is None or delta_auc is None:
        return None
    return float(global_auc - lambda_param * delta_auc)


def per_class_ovr_auc(y_true: np.ndarray, y_proba: np.ndarray, labels: list | None = None) -> dict:
    result = {}
    for cls in range(_canonical_k(y_proba, labels)):
        target = (y_true == cls).astype(np.int8)
        result[f"class_{cls}"] = None if np.unique(target).size != 2 else float(
            roc_auc_score(target, y_proba[:, cls])
        )
    return result


def hand_till_ovo_auc(y_true: np.ndarray, y_proba: np.ndarray, labels: list | None = None) -> list[dict]:
    result = []
    for class_a, class_b in combinations(range(_canonical_k(y_proba, labels)), 2):
        mask = np.isin(y_true, (class_a, class_b))
        y_pair = y_true[mask]
        n_a, n_b = int((y_pair == class_a).sum()), int((y_pair == class_b).sum())
        auc = None
        if n_a and n_b:
            proba = y_proba[mask][:, [class_a, class_b]]
            denom = proba.sum(axis=1)
            valid = denom > 0
            if np.any(valid):
                auc = float(roc_auc_score((y_pair[valid] == class_b).astype(int), proba[valid, 1] / denom[valid]))
        result.append({"class_a": class_a, "class_b": class_b, "auc": auc, "n_a": n_a, "n_b": n_b})
    return result


def k_consistency_check(y_true: np.ndarray, labels: list | None) -> dict:
    observed = np.unique(y_true).astype(int).tolist()
    k_canonical = len(labels) if labels is not None else None
    return {"k_canonical": k_canonical, "k_observed": len(observed),
            "observed_labels": observed,
            "missing_canonical_labels": sorted(set(labels).difference(observed)) if labels is not None else None,
            "matches": None if k_canonical is None else k_canonical == len(observed)}


def null_auc_permutation_test(y_true: np.ndarray, y_proba: np.ndarray, labels: list | None = None,
                              n_perm: int = 1000, seed: int = 42) -> dict:
    rng, result = np.random.default_rng(seed), {}
    for cls in range(_canonical_k(y_proba, labels)):
        target = (y_true == cls).astype(np.int8)
        if np.unique(target).size != 2:
            result[f"class_{cls}"] = None
            continue
        observed = float(roc_auc_score(target, y_proba[:, cls]))
        null = np.asarray([roc_auc_score(rng.permutation(target), y_proba[:, cls]) for _ in range(n_perm)])
        result[f"class_{cls}"] = {"auc_observed": observed, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)), "null_p95": float(np.quantile(null, .95)),
            "p_value_one_sided": float((1 + (null >= observed).sum()) / (n_perm + 1)), "n_permutations": n_perm}
    return result


def chance_adjusted_auc_ba(auc: float | None, balanced_accuracy: float | None, k: int) -> dict:
    if auc is None or balanced_accuracy is None or k < 2:
        return {"auc_relative_to_chance": None, "ba_relative_to_chance": None, "auc_minus_ba_relative": None}
    auc_relative, ba_chance = (auc - .5) / .5, 1.0 / k
    ba_relative = (balanced_accuracy - ba_chance) / (1 - ba_chance)
    return {"auc_relative_to_chance": float(auc_relative), "ba_relative_to_chance": float(ba_relative),
            "auc_minus_ba_relative": float(auc_relative - ba_relative), "auc_chance": .5, "ba_chance": ba_chance}


def compute_auc_ba_diagnostics(y_true: np.ndarray, y_proba: np.ndarray, balanced_accuracy: float | None,
                               labels: list | None = None, n_perm: int = 1000, seed: int = 42) -> dict:
    k = _canonical_k(y_proba, labels)
    try:
        auc = float(roc_auc_score(y_true, y_proba[:, 1])) if k == 2 else float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels))
    except (ValueError, IndexError):
        auc = None
    return {"k_check": k_consistency_check(y_true, labels),
            "per_class_ovr_auc": per_class_ovr_auc(y_true, y_proba, labels),
            "hand_till_ovo_auc": hand_till_ovo_auc(y_true, y_proba, labels),
            "null_auc_permutation": null_auc_permutation_test(y_true, y_proba, labels, n_perm, seed),
            "chance_adjusted_auc_ba": chance_adjusted_auc_ba(auc, balanced_accuracy, k)}
