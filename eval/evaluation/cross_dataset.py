"""
eval/evaluation/cross_dataset.py

Out-of-distribution evaluation: train on one dataset, evaluate on another.

This implements the "inter-dataset evaluation" described in the thesis:
"Se evaluará la capacidad de generalización de los modelos mediante
experimentos entre conjuntos de datos, analizando la degradación del
rendimiento y de las métricas de equidad en escenarios out-of-distribution."
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from eval.evaluation.metrics import compute_metrics, compute_confusion_matrix, compute_error_distribution
from eval.evaluation.fairness import fairness_report, compute_fairness_summary, compute_cross_dataset_fairness_amplification

logger = logging.getLogger(__name__)


def cross_dataset_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_class: Any,
    model_params: dict,
    label_encoder: LabelEncoder,
    embedding_col: str = "embedding",
    sensitive_attrs: list[str] | None = None,
    min_group_size: int = 10,
    seed: int = 42,
) -> dict:
    """
    Train on train_df, evaluate on test_df (out-of-distribution).
    
    Args:
        train_df: Training dataset (from source domain)
        test_df: Test dataset (from target domain, different distribution)
        model_class: Sklearn-compatible model class (e.g., LogisticRegression)
        model_params: Parameters for model initialization
        label_encoder: Fitted LabelEncoder for converting labels
        embedding_col: Column name containing embeddings
        sensitive_attrs: List of sensitive attribute columns for fairness
        min_group_size: Minimum subgroup size for fairness evaluation
        seed: Random seed
        
    Returns:
        dict with train_dataset, test_dataset, metrics, fairness, bootstrap
    """
    from eval.models.linear_probe import stack_embeddings
    from eval.evaluation.bootstrap import bootstrap_metrics, bootstrap_fairness_gaps
    from eval.evaluation.fairness import bin_age
    
    if sensitive_attrs is None:
        sensitive_attrs = ["sex", "age"]
    
    # Extract datasets names
    train_dataset = train_df["dataset"].iloc[0] if "dataset" in train_df.columns else "unknown"
    test_dataset = test_df["dataset"].iloc[0] if "dataset" in test_df.columns else "unknown"
    
    logger.info("Cross-dataset evaluation: %s -> %s", train_dataset, test_dataset)
    
    # Prepare training data
    X_train = stack_embeddings(train_df, embedding_col)
    y_train = label_encoder.transform(train_df["label"])
    
    # Prepare test data
    X_test = stack_embeddings(test_df, embedding_col)
    y_test = label_encoder.transform(test_df["label"])
    
    # Train model
    clf = model_class(**model_params, random_state=seed)
    clf.fit(X_train, y_train)
    
    # Predict on test
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Compute metrics
    test_metrics = compute_metrics(
        y_test, y_pred, y_proba,
        labels=list(range(len(label_encoder.classes_))),
    )
    
    # Confusion matrix and error distribution
    labels = list(range(len(label_encoder.classes_)))
    confusion = compute_confusion_matrix(y_test, y_pred, labels=labels)
    error_dist = compute_error_distribution(y_test, y_pred, labels=labels)
    
    # Bootstrap
    bootstrap_result = bootstrap_metrics(
        y_test, y_pred, y_proba, n_bootstrap=1000, seed=seed
    )
    
    # Fairness evaluation
    fairness_result = fairness_report(
        test_df, y_test, y_pred, y_proba,
        min_group_size=min_group_size, seed=seed,
    )
    
    # Compute fairness summary for each sensitive attribute
    fairness_summary = {}
    for attr in sensitive_attrs:
        if attr in fairness_result:
            subgroup_df = fairness_result[attr]
            # Get global AUC
            auc_col = "auc" if "auc" in test_metrics else "auc_ovr_macro"
            global_auc = test_metrics.get(auc_col)
            
            summary = compute_fairness_summary(
                subgroup_df, global_auc, lambda_param=1.0
            )
            fairness_summary[attr] = summary

    # Bootstrap IC 95% para las brechas de fairness (OOD)
    fairness_gaps_bootstrap = {}
    for attr in fairness_summary:
        if attr == "age" and "age" in test_df.columns:
            gdf = test_df.copy()
            gdf["age_group"] = bin_age(gdf["age"])
            group_col = "age_group"
        elif attr == "sex" and "sex" in test_df.columns:
            gdf = test_df
            group_col = "sex"
        else:
            continue
        if gdf[group_col].notna().sum() < 2:
            continue
        fairness_gaps_bootstrap[attr] = bootstrap_fairness_gaps(
            gdf, y_test, y_pred, y_proba,
            group_col=group_col, min_group_size=min_group_size,
            n_bootstrap=1000, seed=seed,
        )
    
    result = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "metrics": test_metrics,
        "confusion_matrix": confusion,
        "error_distribution": error_dist,
        "bootstrap": bootstrap_result,
        "fairness": fairness_result,
        "fairness_summary": fairness_summary,
        "fairness_gaps_bootstrap": fairness_gaps_bootstrap,
        "predictions": {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": np.asarray(y_proba, dtype=np.float64),
            "uids": test_df["uid"].to_numpy() if "uid" in test_df.columns else None,
        },
    }
    
    logger.info(
        "Cross-dataset %s -> %s: auc=%.3f",
        train_dataset, test_dataset,
        test_metrics.get("auc", test_metrics.get("auc_ovr_macro", 0)),
    )
    
    return result


def run_cross_dataset_evaluation(
    datasets: list[pd.DataFrame],
    model_class: Any,
    model_params: dict,
    label_encoder: LabelEncoder,
    sensitive_attrs: list[str] | None = None,
    min_group_size: int = 10,
    seed: int = 42,
) -> list[dict]:
    """
    Run cross-dataset evaluation for all dataset pairs.
    
    For N datasets, evaluates N*(N-1) pairs (train on i, test on j where i != j).
    
    Args:
        datasets: List of DataFrames, one per dataset
        model_class: Sklearn-compatible model class
        model_params: Parameters for model initialization
        label_encoder: Fitted LabelEncoder
        sensitive_attrs: List of sensitive attribute columns
        min_group_size: Minimum subgroup size for fairness
        seed: Random seed
        
    Returns:
        List of result dicts, one per dataset pair
    """
    results = []
    
    for i, train_df in enumerate(datasets):
        for j, test_df in enumerate(datasets):
            if i == j:
                continue  # Skip same-dataset (in-distribution)
            
            train_name = train_df["dataset"].iloc[0] if "dataset" in train_df.columns else f"dataset_{i}"
            test_name = test_df["dataset"].iloc[0] if "dataset" in test_df.columns else f"dataset_{j}"
            
            logger.info("Cross-dataset: %s -> %s", train_name, test_name)
            
            try:
                result = cross_dataset_evaluate(
                    train_df=train_df,
                    test_df=test_df,
                    model_class=model_class,
                    model_params=model_params,
                    label_encoder=label_encoder,
                    sensitive_attrs=sensitive_attrs,
                    min_group_size=min_group_size,
                    seed=seed,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed cross-dataset %s -> %s: %s", train_name, test_name, e)
                results.append({
                    "train_dataset": train_name,
                    "test_dataset": test_name,
                    "error": str(e),
                })
    
    return results


def compute_ood_degradation(
    in_distribution_results: list[dict],
    cross_dataset_results: list[dict],
) -> pd.DataFrame:
    """
    Compare in-distribution vs out-of-distribution performance.
    
    Args:
        in_distribution_results: Results from in-distribution evaluation
        cross_dataset_results: Results from cross-dataset evaluation
        
    Returns:
        DataFrame with degradation metrics including fairness amplification
    """
    rows = []
    
    for cross_result in cross_dataset_results:
        if "error" in cross_result:
            continue
        
        train_ds = cross_result["train_dataset"]
        test_ds = cross_result["test_dataset"]
        
        # Find corresponding in-distribution result for test dataset
        ind_result = None
        for r in in_distribution_results:
            if r.get("model") == test_ds:
                ind_result = r
                break
        
        if ind_result is None:
            continue
        
        # Compute degradation
        auc_col = "auc" if "auc" in cross_result["metrics"] else "auc_ovr_macro"
        ood_auc = cross_result["metrics"].get(auc_col)
        ind_auc = ind_result.get("test_metrics", {}).get(auc_col)
        
        # Compute balanced accuracy degradation
        ood_ba = cross_result["metrics"].get("balanced_accuracy")
        ind_ba = ind_result.get("test_metrics", {}).get("balanced_accuracy")
        
        # Compute F1 degradation
        ood_f1 = cross_result["metrics"].get("f1_macro")
        ind_f1 = ind_result.get("test_metrics", {}).get("f1_macro")
        
        # Compute fairness amplification
        id_fairness = ind_result.get("fairness_summary", {})
        ood_fairness = cross_result.get("fairness_summary", {})
        fairness_amplification = compute_cross_dataset_fairness_amplification(
            id_fairness, ood_fairness
        )
        
        row = {
            "train_dataset": train_ds,
            "test_dataset": test_ds,
            "ind_auc": ind_auc,
            "ood_auc": ood_auc,
            "auc_degradation": (ind_auc - ood_auc) if (ind_auc is not None and ood_auc is not None) else None,
            "relative_auc_degradation": ((ind_auc - ood_auc) / ind_auc) if (ind_auc is not None and ood_auc is not None and ind_auc > 0) else None,
            "ind_balanced_accuracy": ind_ba,
            "ood_balanced_accuracy": ood_ba,
            "ba_degradation": (ind_ba - ood_ba) if (ind_ba is not None and ood_ba is not None) else None,
            "ind_f1_macro": ind_f1,
            "ood_f1_macro": ood_f1,
            "f1_degradation": (ind_f1 - ood_f1) if (ind_f1 is not None and ood_f1 is not None) else None,
        }
        
        # Add fairness amplification metrics
        for attr, amp in fairness_amplification.items():
            row[f"{attr}_delta_auc_amplified"] = amp["delta_auc_amplified"]
            row[f"{attr}_delta_auc_change"] = amp["delta_auc_change"]
            row[f"{attr}_avg_gap_change"] = amp["avg_gap_change"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Ensamblado BA-ID / BA-OOD / ΔBA  (Cambio 2 del pipeline de generalización)
# ─────────────────────────────────────────────────────────────────────────────
# BA-ID  : métricas del JSON de evaluación del ORIGEN con CV-OOF (100% de la
#          cohorte) una vez aplicado el Cambio 1.
# BA-OOD : métricas del JSON cross_{orig}_to_{dest} (modelo 100%-data del
#          origen contra el dataset externo; no cambia con Cambio 1).
# ΔBA    = BA_ID - BA_OOD, con IC 95% por bootstrap de la diferencia
#          (remuestreo independiente de pacientes de ambas poblaciones).

_METRIC_SCORERS = {
    "balanced_accuracy": lambda y_true, y_pred: float(balanced_accuracy_score(y_true, y_pred)),
    "f1_macro": lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
}


def _ba_f1_point(y_true, y_pred):
    return {
        "balanced_accuracy": _METRIC_SCORERS["balanced_accuracy"](y_true, y_pred),
        "f1_macro": _METRIC_SCORERS["f1_macro"](y_true, y_pred),
    }


def bootstrap_delta_metrics(
    id_pred: dict,
    ood_pred: dict,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    metric_names: list[str] | None = None,
) -> dict:
    """
    IC 95% por bootstrap de ΔBA / ΔF1 entre BA-ID y BA-OOD.

    Cada iteración remuestra con reemplazo los pacientes de la población ID
    (predicciones OOF combinadas de los 5 folds) y de la población OOD
    (predicciones del modelo 100%-data sobre el target) de forma
    INDEPENDIENTE, recalcula la métrica en cada una y resta:
        delta = metrica_ID - metrica_OOD.

    Args:
        id_pred:  dict con "y_true"/"y_pred" (serie ID, típicamente OOF).
        ood_pred: dict con "y_true"/"y_pred" (serie OOD sobre el target).
        metric_names: métricas a bootstrappear (balanced_accuracy, f1_macro).

    Returns:
        dict con punto + CI por métrica, con claves point/ci_low/ci_high/
        n_valid_bootstrap, replicando el formato de bootstrap_metrics().
    """
    metric_names = metric_names or ["balanced_accuracy", "f1_macro"]
    scorers = {k: _METRIC_SCORERS[k] for k in metric_names if k in _METRIC_SCORERS}
    if not scorers:
        raise ValueError(f"ninguna métrica válida en {metric_names}")

    id_yt = np.asarray(id_pred["y_true"])
    id_yp = np.asarray(id_pred["y_pred"])
    ood_yt = np.asarray(ood_pred["y_true"])
    ood_yp = np.asarray(ood_pred["y_pred"])
    n_id, n_ood = len(id_yt), len(ood_yt)

    rng = np.random.default_rng(seed)
    samples = {name: [] for name in scorers}
    n_failed = 0

    for _ in range(n_bootstrap):
        i_id = rng.integers(0, n_id, size=n_id)
        i_ood = rng.integers(0, n_ood, size=n_ood)
        try:
            vals = {
                name: fn(id_yt[i_id], id_yp[i_id]) - fn(ood_yt[i_ood], ood_yp[i_ood])
                for name, fn in scorers.items()
            }
        except Exception:
            n_failed += 1
            continue
        for name, v in vals.items():
            samples[name].append(float(v))

    alpha = (1 - ci) / 2
    point_id = _ba_f1_point(id_yt, id_yp)
    point_ood = _ba_f1_point(ood_yt, ood_yp)

    result = {}
    for name in scorers:
        arr = np.asarray(samples[name])
        point = float(point_id[name] - point_ood[name])
        if len(arr) == 0:
            result[name] = {"point": point, "ci_low": None, "ci_high": None, "n_valid_bootstrap": 0}
        else:
            result[name] = {
                "point": point,
                "ci_low": float(np.percentile(arr, 100 * alpha)),
                "ci_high": float(np.percentile(arr, 100 * (1 - alpha))),
                "n_valid_bootstrap": len(arr),
            }

    result["_n_bootstrap_requested"] = n_bootstrap
    result["_n_bootstrap_failed"] = n_failed
    return result


def assemble_ba_degradation(
    id_metrics: dict,
    id_bootstrap: dict,
    ood_metrics: dict,
    ood_bootstrap: dict,
    delta_bootstrap: dict,
) -> dict:
    """
    Arma el bloque fila BA-ID / BA-OOD / ΔBA / ΔF1 (con IC 95%) para una
    combinación origen → destino, a partir de los JSONs de run_eval y del
    bootstrap de la diferencia calculado en bootstrap_delta_metrics().

    id_metrics/ood_metrics: test_metrics del JSON (contienen balanced_accuracy
        y f1_macro).
    id_bootstrap/ood_bootstrap: bootstrap del JSON (IC por métrica).
    delta_bootstrap: salida de bootstrap_delta_metrics().

    Retorna un dict listo para fila de tabla (puntos + IC formateados).
    """
    def _ci_of(bootstrap: dict, metric: str) -> str:
        entry = bootstrap.get(metric) or {}
        lo, hi = entry.get("ci_low"), entry.get("ci_high")
        if lo is None or hi is None:
            return "—"
        return f"({lo:.3f}, {hi:.3f})"

    def _delta_of(delta: dict, metric: str) -> tuple[float | None, str]:
        entry = delta.get(metric) or {}
        point = entry.get("point")
        lo, hi = entry.get("ci_low"), entry.get("ci_high")
        if point is None:
            return None, "—"
        if lo is None or hi is None:
            return point, "—"
        return point, f"({lo:.3f}, {hi:.3f})"

    ba_id = id_metrics.get("balanced_accuracy")
    f1_id = id_metrics.get("f1_macro")
    ba_ood = ood_metrics.get("balanced_accuracy")
    f1_ood = ood_metrics.get("f1_macro")

    delta_ba, delta_ba_ci = _delta_of(delta_bootstrap, "balanced_accuracy")
    delta_f1, delta_f1_ci = _delta_of(delta_bootstrap, "f1_macro")

    return {
        "ba_id": ba_id,
        "ba_id_ci": _fmt_ci(id_bootstrap, "balanced_accuracy"),
        "f1_id": f1_id,
        "f1_id_ci": _fmt_ci(id_bootstrap, "f1_macro"),
        "ba_ood": ba_ood,
        "ba_ood_ci": _fmt_ci(ood_bootstrap, "balanced_accuracy"),
        "f1_ood": f1_ood,
        "f1_ood_ci": _fmt_ci(ood_bootstrap, "f1_macro"),
        "delta_ba": delta_ba,
        "delta_ba_ci": delta_ba_ci,
        "delta_f1": delta_f1,
        "delta_f1_ci": delta_f1_ci,
    }


def _fmt_ci(bootstrap: dict, metric: str) -> str:
    entry = bootstrap.get(metric) or {}
    lo, hi = entry.get("ci_low"), entry.get("ci_high")
    if lo is None or hi is None:
        return "—"
    return f"({lo:.3f}, {hi:.3f})"