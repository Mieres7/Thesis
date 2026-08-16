"""
eval/evaluation/fairness.py

Computes classification metrics separately per demographic subgroup
(sex, age bucket) to detect performance disparities across groups.

Includes fairness metrics (ΔAUC, Equalized Odds) and trade-off metrics
(AUC_ES) as defined in the thesis evaluation framework.

Design notes:
    - Age is continuous, so it is bucketed via `bin_age()` before grouping.
      Default: median split (two roughly balanced groups). Pass explicit
      `bins`/`labels` for a clinically meaningful cutoff (e.g. menopause
      age ~50) instead of the data-driven median.
    - Reuses compute_metrics() / bootstrap_metrics() so fairness numbers
      are computed exactly the same way as the overall metrics -- no
      separate metric logic to keep in sync.
    - Subgroups with too few patients (< min_group_size) are flagged
      instead of silently producing an unstable/misleading metric.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from eval.evaluation.metrics import (
    compute_metrics,
    compute_tpr_fpr,
    compute_delta_auc,
    compute_delta_ba,
    compute_delta_f1,
    compute_average_gap,
    compute_equalized_odds_gap,
    compute_auc_es,
    compute_confusion_matrix,
    compute_error_distribution,
)
from eval.evaluation.bootstrap import bootstrap_metrics

logger = logging.getLogger(__name__)


DEFAULT_AGE_BINS: list[float] = [0, 50, 70, 200]
DEFAULT_AGE_LABELS: list[str] = ["<50", "50-70", ">=70"]


def bin_age(
    age: pd.Series,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.Series:
    """
    Bucket a continuous age column into groups.

    Default: clinically motivated 3-group split for breast cancer:
        <50  (premenopausal)
        50-70 (postmenopausal standard)
        >=70 (elderly)

    Override with any custom bins, e.g.:
        bin_age(age, bins=[0, 45, 65, 200], labels=["<45", "45-65", ">=65"])
    """
    age_clean = pd.to_numeric(age, errors="coerce")
    if bins is None:
        bins = DEFAULT_AGE_BINS
    if labels is None:
        labels = DEFAULT_AGE_LABELS
    return pd.cut(age_clean, bins=bins, labels=labels)


def evaluate_by_subgroup(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    group_col: str,
    min_group_size: int = 10,
    with_bootstrap: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute metrics separately for each value of `group_col` (e.g. "sex"
    or a binned age column already present in df).

    df, y_true, y_pred, y_proba must all be aligned (same order, same
    length) -- typically the test set after predict()/predict_proba().
    """
    groups = df[group_col].values
    unique_groups = pd.unique(groups[~pd.isna(groups)])

    rows = []
    for g in unique_groups:
        mask = df[group_col].eq(g).to_numpy()
        n = mask.sum()

        if n < min_group_size:
            logger.warning(
                "[%s=%s] only %d patients (< min_group_size=%d) -- "
                "metrics may be unstable, flagged as low_n",
                group_col, g, n, min_group_size,
            )

        y_true_g, y_pred_g, y_proba_g = y_true[mask], y_pred[mask], y_proba[mask]

        if len(np.unique(y_true_g)) < 2:
            logger.warning(
                "[%s=%s] only one class present (n=%d) -- cannot compute AUC",
                group_col, g, n,
            )
            row = {"group": g, "n": n, "low_n": n < min_group_size}
        elif with_bootstrap:
            bs_result = bootstrap_metrics(
                y_true_g, y_pred_g, y_proba_g, n_bootstrap=n_bootstrap, seed=seed
            )
            row = {"group": g, "n": n, "low_n": n < min_group_size}
            for metric_name, vals in bs_result.items():
                if metric_name.startswith("_"):
                    continue
                row[f"{metric_name}_point"] = vals["point"]
                row[f"{metric_name}_ci_low"] = vals["ci_low"]
                row[f"{metric_name}_ci_high"] = vals["ci_high"]
            # Add TPR/FPR for fairness metrics
            tpr_fpr = compute_tpr_fpr(y_true_g, y_pred_g)
            row["tpr"] = tpr_fpr["tpr"]
            row["fpr"] = tpr_fpr["fpr"]
        else:
            m = compute_metrics(y_true_g, y_pred_g, y_proba_g)
            # Add TPR/FPR for fairness metrics
            tpr_fpr = compute_tpr_fpr(y_true_g, y_pred_g)
            row = {"group": g, "n": n, "low_n": n < min_group_size, **m, **tpr_fpr}

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    return result


def fairness_report(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    age_bins: list[float] | None = None,
    age_labels: list[str] | None = None,
    min_group_size: int = 10,
    with_bootstrap: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper: runs evaluate_by_subgroup() for sex and binned age
    in one call. Returns a dict {"sex": df, "age": df}.

    Skips a dimension gracefully (with a log message) if the corresponding
    column is entirely missing/NaN -- e.g. datasets where `sex` was not
    available yet.
    """
    report = {}

    if "sex" in df.columns and df["sex"].notna().any():
        report["sex"] = evaluate_by_subgroup(
            df, y_true, y_pred, y_proba, group_col="sex",
            min_group_size=min_group_size, with_bootstrap=with_bootstrap,
            n_bootstrap=n_bootstrap, seed=seed,
        )
    else:
        logger.warning("Skipping sex fairness: column missing or all-NaN")

    if "age" in df.columns and df["age"].notna().any():
        df = df.copy()
        df["age_group"] = bin_age(df["age"], bins=age_bins, labels=age_labels)
        report["age"] = evaluate_by_subgroup(
            df, y_true, y_pred, y_proba, group_col="age_group",
            min_group_size=min_group_size, with_bootstrap=with_bootstrap,
            n_bootstrap=n_bootstrap, seed=seed,
        )
    else:
        logger.warning("Skipping age fairness: column missing or all-NaN")

    return report


def compute_fairness_summary(
    subgroup_results: pd.DataFrame,
    global_auc: float,
    lambda_param: float = 1.0,
) -> dict:
    """
    Compute fairness summary metrics from subgroup evaluation results.
    
    Computes:
        - ΔBA: maximum balanced accuracy difference between subgroups
        - ΔF1: maximum macro-F1 difference between subgroups
        - ΔAUC: maximum AUC difference between subgroups
        - Equalized Odds gaps: maximum TPR and FPR differences
        - AUC_ES: equity-scaled AUC
    
    Args:
        subgroup_results: DataFrame from evaluate_by_subgroup() with columns
                         including 'group', 'balanced_accuracy', 'f1_macro',
                         'auc' (or 'auc_ovr_macro'), 'tpr', 'fpr'
        global_auc: Overall AUC across all groups
        lambda_param: Trade-off parameter for AUC_ES (higher = more penalty for unfairness)
        
    Returns:
        dict with delta_ba, delta_f1, delta_auc, equalized_odds, auc_es
    """
    # Extract AUC values per subgroup
    auc_col = "auc" if "auc" in subgroup_results.columns else "auc_ovr_macro"
    subgroup_auc = {}
    subgroup_tpr = {}
    subgroup_fpr = {}
    subgroup_ba = {}
    subgroup_f1 = {}
    
    for _, row in subgroup_results.iterrows():
        group = row["group"]
        if pd.notna(row.get(auc_col)):
            subgroup_auc[group] = row[auc_col]
        if pd.notna(row.get("tpr")):
            subgroup_tpr[group] = row["tpr"]
        if pd.notna(row.get("fpr")):
            subgroup_fpr[group] = row["fpr"]
        if pd.notna(row.get("balanced_accuracy")):
            subgroup_ba[group] = row["balanced_accuracy"]
        if pd.notna(row.get("f1_macro")):
            subgroup_f1[group] = row["f1_macro"]
    
    # Compute ΔBA and ΔF1 (performance gaps)
    delta_ba_result = compute_delta_ba(subgroup_ba)
    delta_f1_result = compute_delta_f1(subgroup_f1)
    
    # Compute ΔAUC
    delta_auc_result = compute_delta_auc(subgroup_auc)
    
    # Compute average gap
    avg_gap_result = compute_average_gap(subgroup_auc, global_auc)
    
    # Compute Equalized Odds gaps
    equalized_odds_result = compute_equalized_odds_gap(subgroup_tpr, subgroup_fpr)
    
    # Compute AUC_ES
    auc_es = compute_auc_es(global_auc, delta_auc_result["delta_auc"], lambda_param)
    
    return {
        "delta_ba": delta_ba_result,
        "delta_f1": delta_f1_result,
        "delta_auc": delta_auc_result,
        "avg_gap": avg_gap_result,
        "equalized_odds": equalized_odds_result,
        "auc_es": auc_es,
        "n_subgroups": len(subgroup_auc),
    }


def compute_error_analysis_by_subgroup(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    labels: list | None = None,
    min_group_size: int = 10,
) -> dict:
    """
    Compute error distribution for each subgroup.

    Args:
        df: DataFrame with group column
        y_true: True labels (aligned with df)
        y_pred: Predicted labels (aligned with df)
        group_col: Column name for grouping (e.g., "sex", "age_group")
        labels: Class labels for confusion matrix
        min_group_size: Minimum subgroup size to include

    Returns:
        dict mapping group_name -> error analysis dict
    """
    groups = df[group_col].values
    unique_groups = pd.unique(groups[~pd.isna(groups)])

    result = {}
    for g in unique_groups:
        mask = df[group_col].eq(g).to_numpy()
        n = mask.sum()

        if n < min_group_size:
            continue

        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        if len(np.unique(y_true_g)) < 2:
            continue

        error_dist = compute_error_distribution(y_true_g, y_pred_g, labels=labels)
        cm = compute_confusion_matrix(y_true_g, y_pred_g, labels=labels)

        result[str(g)] = {
            "n": int(n),
            "error_rate": error_dist["error_rate"],
            "total_errors": error_dist["total_errors"],
            "most_confused": error_dist["most_confused"],
            "confusion_matrix": cm["confusion_matrix"],
            "per_class": cm["per_class"],
        }

    return result


def compute_cross_dataset_fairness_amplification(
    id_fairness_summary: dict,
    ood_fairness_summary: dict,
) -> dict:
    """
    Compare fairness metrics between in-distribution (ID) and out-of-distribution (OOD)
    scenarios to detect fairness gap amplification.

    As defined in the thesis:
        "se evaluará si las brechas de equidad se amplían al cambiar de distribución"

    Args:
        id_fairness_summary: Fairness summary from in-distribution evaluation
        ood_fairness_summary: Fairness summary from out-of-distribution evaluation

    Returns:
        dict with amplification metrics per sensitive attribute
    """
    result = {}

    for attr in set(list(id_fairness_summary.keys()) + list(ood_fairness_summary.keys())):
        id_summary = id_fairness_summary.get(attr, {})
        ood_summary = ood_fairness_summary.get(attr, {})

        id_delta = id_summary.get("delta_auc", {}).get("delta_auc")
        ood_delta = ood_summary.get("delta_auc", {}).get("delta_auc")

        id_avg_gap = id_summary.get("avg_gap", {}).get("avg_gap")
        ood_avg_gap = ood_summary.get("avg_gap", {}).get("avg_gap")

        id_tpr_gap = id_summary.get("equalized_odds", {}).get("tpr_gap")
        ood_tpr_gap = ood_summary.get("equalized_odds", {}).get("tpr_gap")

        id_fpr_gap = id_summary.get("equalized_odds", {}).get("fpr_gap")
        ood_fpr_gap = ood_summary.get("equalized_odds", {}).get("fpr_gap")

        amplification = {
            "delta_auc_id": id_delta,
            "delta_auc_ood": ood_delta,
            "delta_auc_change": (ood_delta - id_delta) if (id_delta is not None and ood_delta is not None) else None,
            "delta_auc_amplified": (ood_delta > id_delta) if (id_delta is not None and ood_delta is not None) else None,

            "avg_gap_id": id_avg_gap,
            "avg_gap_ood": ood_avg_gap,
            "avg_gap_change": (ood_avg_gap - id_avg_gap) if (id_avg_gap is not None and ood_avg_gap is not None) else None,

            "tpr_gap_id": id_tpr_gap,
            "tpr_gap_ood": ood_tpr_gap,
            "tpr_gap_change": (ood_tpr_gap - id_tpr_gap) if (id_tpr_gap is not None and ood_tpr_gap is not None) else None,

            "fpr_gap_id": id_fpr_gap,
            "fpr_gap_ood": ood_fpr_gap,
            "fpr_gap_change": (ood_fpr_gap - id_fpr_gap) if (id_fpr_gap is not None and ood_fpr_gap is not None) else None,
        }

        result[attr] = amplification

    return result
