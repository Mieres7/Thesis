"""
eval/evaluation/bootstrap.py

Patient-level bootstrap confidence intervals for classification metrics.

Resampling is done on unique `uid` values (not on rows), so this is safe
even if pooling strategy ever changes to keep multiple rows per patient.
Since y_true/y_pred/y_proba here are already patient-level arrays (one
entry per patient, produced after embedding_loader's pooling), bootstrap
resampling with replacement on indices IS resampling on patients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.evaluation.metrics import compute_metrics


def _is_nan(v) -> bool:
    if v is None:
        return False
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Resample patients with replacement n_bootstrap times, recompute metrics
    each time, and return point estimate + percentile CI for each metric.

    Incluye intervalos de confianza para la sensibilidad (recall) por clase,
    bajo la clave ``per_class_sensitivity``.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point_estimate = compute_metrics(y_true, y_pred, y_proba)
    metric_names = [k for k, v in point_estimate.items() if isinstance(v, (int, float))]

    class_sens = point_estimate.get("per_class_sensitivity", {}) or {}
    class_names = list(class_sens.keys())

    samples = {name: [] for name in metric_names}
    class_samples = {name: [] for name in class_names}
    n_failed = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_bs = y_true[idx]

        if len(np.unique(y_true_bs)) < 2:
            n_failed += 1
            continue

        m = compute_metrics(y_true_bs, y_pred[idx], y_proba[idx])
        if any(_is_nan(m.get(name)) for name in metric_names):
            n_failed += 1
            continue
        for name in metric_names:
            val = m.get(name)
            if val is not None:
                samples[name].append(val)

        m_class = m.get("per_class_sensitivity", {}) or {}
        for name in class_names:
            val = m_class.get(name)
            if val is not None and not _is_nan(val):
                class_samples[name].append(val)

    alpha = 1 - ci
    result = {}
    for name in metric_names:
        vals = np.array(samples[name])
        if len(vals) == 0:
            result[name] = {
                "point": point_estimate[name], "ci_low": None, "ci_high": None,
                "n_valid_bootstrap": 0,
            }
            continue
        result[name] = {
            "point": point_estimate[name],
            "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_valid_bootstrap": len(vals),
        }

    # IC 95% para sensibilidad (recall) por clase
    if class_names:
        class_result = {}
        for name in class_names:
            vals = np.array(class_samples[name])
            point = class_sens.get(name)
            if len(vals) == 0:
                class_result[name] = {
                    "point": point, "ci_low": None, "ci_high": None,
                    "n_valid_bootstrap": 0,
                }
                continue
            class_result[name] = {
                "point": point,
                "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
                "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
                "n_valid_bootstrap": len(vals),
            }
        result["per_class_sensitivity"] = class_result

    result["_n_bootstrap_requested"] = n_bootstrap
    result["_n_bootstrap_failed"] = n_failed
    return result


def bootstrap_fairness_gaps(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    group_col: str,
    min_group_size: int = 10,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap IC 95% para las brechas de fairness (ΔBA, ΔF1, ΔAUC, gap
    promedio, gaps TPR/FPR y AUC-ES) remuestreando pacientes con reemplazo.

    Cada remuestreo recalcula las métricas por subgrupo y el resumen de
    fairness completo, de modo que los IC reflejan la variación muestral
    de las brechas (no solo de las métricas globales).
    """
    from eval.evaluation.fairness import evaluate_by_subgroup, compute_fairness_summary

    rng = np.random.default_rng(seed)
    n = len(y_true)

    groups = df[group_col].to_numpy()
    global_auc = compute_metrics(y_true, y_pred, y_proba)
    auc_key = "auc" if "auc" in global_auc else "auc_ovr_macro"

    gap_names = ["delta_ba", "delta_f1", "delta_auc", "avg_gap", "tpr_gap", "fpr_gap", "auc_es"]
    samples = {name: [] for name in gap_names}
    n_failed = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        gr = groups[idx]

        if len(np.unique(yt)) < 2:
            n_failed += 1
            continue

        sub_df = pd.DataFrame({group_col: gr})
        try:
            subgroup = evaluate_by_subgroup(
                sub_df, yt, yp, ypr, group_col, min_group_size=min_group_size,
            )
            g_auc = compute_metrics(yt, yp, ypr).get(auc_key)
            fs = compute_fairness_summary(subgroup, g_auc, lambda_param=1.0)
        except Exception:
            n_failed += 1
            continue

        vals = {
            "delta_ba": (fs.get("delta_ba") or {}).get("delta_ba"),
            "delta_f1": (fs.get("delta_f1") or {}).get("delta_f1"),
            "delta_auc": (fs.get("delta_auc") or {}).get("delta_auc"),
            "avg_gap": (fs.get("avg_gap") or {}).get("avg_gap"),
            "tpr_gap": (fs.get("equalized_odds") or {}).get("tpr_gap"),
            "fpr_gap": (fs.get("equalized_odds") or {}).get("fpr_gap"),
            "auc_es": fs.get("auc_es"),
        }
        recorded = False
        for name in gap_names:
            v = vals[name]
            if v is None or _is_nan(v):
                continue
            samples[name].append(v)
            recorded = True
        if not recorded:
            n_failed += 1

    # Estimación puntual sobre los datos completos
    subgroup_full = evaluate_by_subgroup(
        df, y_true, y_pred, y_proba, group_col, min_group_size=min_group_size,
    )
    full_fs = compute_fairness_summary(subgroup_full, global_auc.get(auc_key), lambda_param=1.0)
    point_vals = {
        "delta_ba": (full_fs.get("delta_ba") or {}).get("delta_ba"),
        "delta_f1": (full_fs.get("delta_f1") or {}).get("delta_f1"),
        "delta_auc": (full_fs.get("delta_auc") or {}).get("delta_auc"),
        "avg_gap": (full_fs.get("avg_gap") or {}).get("avg_gap"),
        "tpr_gap": (full_fs.get("equalized_odds") or {}).get("tpr_gap"),
        "fpr_gap": (full_fs.get("equalized_odds") or {}).get("fpr_gap"),
        "auc_es": full_fs.get("auc_es"),
    }

    alpha = 1 - ci
    result = {}
    for name in gap_names:
        vals = np.array(samples[name])
        if len(vals) == 0:
            result[name] = {
                "point": point_vals[name], "ci_low": None, "ci_high": None,
                "n_valid_bootstrap": 0,
            }
            continue
        result[name] = {
            "point": point_vals[name],
            "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_valid_bootstrap": len(vals),
        }

    result["_n_bootstrap_requested"] = n_bootstrap
    result["_n_bootstrap_failed"] = n_failed
    return result
