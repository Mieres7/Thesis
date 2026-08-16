"""
eval/evaluation/center_effect.py

Cuantificación del center effect: clasificador de ORIGEN de dataset sobre
embeddings a nivel de paciente, reutilizando el protocolo experimental del
Capítulo 6 (MLP probe de 256 unidades + ReLU, alpha=1e-4, max_iter=500,
sobremuestreo de clases minoritarias, StratifiedGroupKFold k=5 agrupado por
uid, seed=42, bootstrap de 1000 remuestreos a nivel de paciente para IC 95%).

- Experimento A: clasificador multiclase por dominio (3 datasets por dominio),
  etiqueta objetivo = dataset de origen (no la etiqueta clínica).
- Experimento B: pares binarios en colorrectal para aislar el efecto escáner:
    B1 vs B2 (mismo escáner) / SurGen vs B1 / SurGen vs B2 (escáner distinto).

NO re-extrae parches ni embeddings: consume los .pkl consolidados que ya
genera el pipeline (final_df_{model}_{task}.pkl).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from eval.evaluation.bootstrap import bootstrap_metrics
from eval.evaluation.metrics import compute_confusion_matrix, compute_metrics
from eval.evaluation.splitter import cross_validate_small_dataset
from eval.models.linear_probe import stack_embeddings
from eval.models.mlp_probe import fit_mlp_probe

logger = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 1000
N_SPLITS = 5

# Nombres de dataset en los .pkl consolidados -> etiqueta de presentación
DATASET_DISPLAY = {
    "BCNB": "BCNB",
    "HISTAI_Breast": "HISTAI-Breast",
    "HistologyHSI": "HSI-BC",
    "HISTAI_CRC_B1": "HISTAI-CRC-B1",
    "HISTAI_CRC_B2": "HISTAI-CRC-B2",
    "SurGen": "SurGen",
}

DOMAIN_DISPLAY = {
    "breast": "Mama",
    "colorectal": "Colorrectal",
}

# Experimento A — multiclase por dominio (etiqueta = dataset de origen)
EXPERIMENT_A = [
    {
        "experiment": "A",
        "key": "A_breast",
        "domain": "breast",
        "task_key": "breast_molsub",
        "datasets": ["BCNB", "HISTAI_Breast", "HistologyHSI"],
    },
    {
        "experiment": "A",
        "key": "A_colorectal",
        "domain": "colorectal",
        "task_key": "crc_site",
        "datasets": ["SurGen", "HISTAI_CRC_B1", "HISTAI_CRC_B2"],
    },
]

# Experimento B — pares binarios en colorrectal (aislar el escáner)
EXPERIMENT_B = [
    {
        "experiment": "B",
        "key": "B_b1_vs_b2",
        "domain": "colorectal",
        "task_key": "crc_site",
        "datasets": ["HISTAI_CRC_B1", "HISTAI_CRC_B2"],
        "note": "mismo escáner, mismo dominio -> señal de sitio/institución",
    },
    {
        "experiment": "B",
        "key": "B_surgen_vs_b1",
        "domain": "colorectal",
        "task_key": "crc_site",
        "datasets": ["SurGen", "HISTAI_CRC_B1"],
        "note": "escáner distinto",
    },
    {
        "experiment": "B",
        "key": "B_surgen_vs_b2",
        "domain": "colorectal",
        "task_key": "crc_site",
        "datasets": ["SurGen", "HISTAI_CRC_B2"],
        "note": "escáner distinto",
    },
]


def comparison_label(datasets: list[str]) -> str:
    """'SurGen vs HISTAI-CRC-B1 vs HISTAI-CRC-B2' a partir de los keys."""
    return " vs ".join(DATASET_DISPLAY.get(d, d) for d in datasets)


def _align_proba_to_full_classes(
    clf,
    y_proba: np.ndarray,
    n_full_classes: int,
) -> np.ndarray:
    """
    Alinea predict_proba() al conjunto de clases completo.

    Con StratifiedGroupKFold cada fold debería ver todas las clases, pero si
    una clase rara no aparece en el train de un fold, el proba de ese fold
    tiene menos columnas; se rellenan con 0 para poder apilar con vstack.
    """
    if y_proba.shape[1] == n_full_classes:
        return y_proba
    aligned = np.zeros((y_proba.shape[0], n_full_classes))
    clf_classes = np.asarray(clf.classes_, dtype=int)
    aligned[:, clf_classes] = y_proba
    return aligned


def center_effect_classify(
    df: pd.DataFrame,
    datasets: list[str],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    n_bootstrap: int = N_BOOTSTRAP,
    label_col: str = "dataset",
) -> dict:
    """
    Clasificador de origen de dataset (MLP probe) con 5-fold CV agrupado por
    paciente (uid). Predicciones OOF combinadas -> 100% de pacientes evaluados.

    El protocolo replica el de la evaluación ID del Capítulo 6:
        - MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4, max_iter=500)
        - sobremuestreo de clases minoritarias (fit_mlp_probe balanced=True)
        - StratifiedGroupKFold(k=5, shuffle=True, random_state=seed) por uid
        - bootstrap de n_bootstrap remuestreos a nivel de paciente (IC 95%)

    Returns:
        dict con n_total/n_test, classes, test_metrics (BA, macro-F1, AUC
        one-vs-rest para multiclase / ROC AUC para binario), confusion_matrix,
        bootstrap (IC 95%) y per_fold (métricas por fold).
    """
    sub = df[df["dataset"].isin(datasets)].copy().reset_index(drop=True)
    if sub["dataset"].nunique() < 2:
        raise ValueError(
            f"Se necesitan >= 2 datasets para el clasificador de origen, "
            f"encontrados: {sorted(sub['dataset'].unique().tolist())}"
        )
    logger.info(
        "Center effect: datasets=%s n=%d (por clase: %s)",
        datasets, len(sub), dict(sub["dataset"].value_counts()),
    )

    # Etiqueta objetivo = dataset de origen (no tocar la etiqueta clínica)
    sub["ce_label"] = sub[label_col]

    # Fit sobre el dataset COMPLETO (misma convención que el CV del pipeline:
    # los folds podrían perder clases raras y transform() fallaría)
    label_encoder = LabelEncoder()
    label_encoder.fit(sub["ce_label"])

    folds = cross_validate_small_dataset(
        sub, n_splits=n_splits, seed=seed, label_col="ce_label",
    )

    n_full_classes = len(label_encoder.classes_)
    all_y_true, all_y_pred, all_y_proba, all_uids = [], [], [], []
    per_fold = []

    for fold_idx in sorted(folds.keys()):
        fold = folds[fold_idx]
        train_df, test_df = fold["train"], fold["test"]

        X_train = stack_embeddings(train_df)
        y_train = label_encoder.transform(train_df["ce_label"])
        clf = fit_mlp_probe(X_train, y_train, seed=seed)

        X_test = stack_embeddings(test_df)
        y_test = label_encoder.transform(test_df["ce_label"])
        y_pred = clf.predict(X_test)
        y_proba = _align_proba_to_full_classes(
            clf, clf.predict_proba(X_test), n_full_classes,
        )

        fold_metrics = compute_metrics(
            y_test, y_pred, y_proba, labels=list(range(n_full_classes)),
        )
        per_fold.append({
            "fold": int(fold_idx),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "classes_in_train": label_encoder.classes_[np.unique(y_train)].tolist(),
            "classes_in_test": label_encoder.classes_[np.unique(y_test)].tolist(),
            "metrics": fold_metrics,
        })
        logger.info(
            "[fold %d/%d] train=%d test=%d ba=%.3f",
            fold_idx + 1, n_splits, len(train_df), len(test_df),
            fold_metrics.get("balanced_accuracy", float("nan")),
        )

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        all_y_proba.append(y_proba)
        all_uids.append(test_df["uid"].to_numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_proba = np.vstack(all_y_proba)

    labels = list(range(n_full_classes))
    test_metrics = compute_metrics(y_true, y_pred, y_proba, labels=labels)
    confusion = compute_confusion_matrix(y_true, y_pred, labels=labels)
    bootstrap = bootstrap_metrics(
        y_true, y_pred, y_proba, n_bootstrap=n_bootstrap, seed=seed,
    )

    logger.info(
        "Center effect OOF: n_test=%d ba=%.3f f1=%.3f auc=%s",
        len(y_true),
        test_metrics.get("balanced_accuracy", float("nan")),
        test_metrics.get("f1_macro", float("nan")),
        test_metrics.get("auc", test_metrics.get("auc_ovr_macro")),
    )

    return {
        "n_total": int(len(sub)),
        "n_test": int(len(y_true)),
        "classes": label_encoder.classes_.tolist(),
        "test_metrics": test_metrics,
        "confusion_matrix": confusion,
        "bootstrap": {k: v for k, v in bootstrap.items() if not k.startswith("_")},
        "per_fold": per_fold,
        "predictions": {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": np.asarray(y_proba, dtype=np.float64),
            "uids": np.concatenate(all_uids),
        },
    }


def auc_from_metrics(metrics: dict) -> tuple[str, float | None]:
    """(clave, valor) de AUC: 'auc' binario / 'auc_ovr_macro' multiclase."""
    if "auc" in metrics:
        return "auc", metrics.get("auc")
    if "auc_ovr_macro" in metrics:
        return "auc_ovr_macro", metrics.get("auc_ovr_macro")
    return "auc", None


def ci_string(bootstrap: dict, metric: str) -> str:
    """'0.835 [0.818, 0.852]' o '0.835 [—]' si no hay IC válido."""
    entry = bootstrap.get(metric) or {}
    point = entry.get("point")
    lo, hi = entry.get("ci_low"), entry.get("ci_high")
    if point is None:
        return "—"
    if lo is None or hi is None:
        return f"{point:.3f} [—]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def summary_row(exp: dict, model: str, result: dict) -> dict:
    """Fila del resumen consolidado (CSV + tabla) para una combinación."""
    metrics = result["test_metrics"]
    bootstrap = result["bootstrap"]
    auc_key, auc = auc_from_metrics(metrics)

    return {
        "experiment": exp["experiment"],
        "domain": exp["domain"],
        "domain_display": DOMAIN_DISPLAY.get(exp["domain"], exp["domain"]),
        "comparison": comparison_label(exp["datasets"]),
        "model": model,
        "n_test": result["n_test"],
        "n_total": result["n_total"],
        "classes": "|".join(result["classes"]),
        "ba": metrics.get("balanced_accuracy"),
        "ba_ci_low": (bootstrap.get("balanced_accuracy") or {}).get("ci_low"),
        "ba_ci_high": (bootstrap.get("balanced_accuracy") or {}).get("ci_high"),
        "ba_ci": ci_string(bootstrap, "balanced_accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_ci_low": (bootstrap.get("f1_macro") or {}).get("ci_low"),
        "f1_ci_high": (bootstrap.get("f1_macro") or {}).get("ci_high"),
        "f1_ci": ci_string(bootstrap, "f1_macro"),
        "auc_metric": auc_key,
        "auc": auc,
        "auc_ci_low": (bootstrap.get(auc_key) or {}).get("ci_low"),
        "auc_ci_high": (bootstrap.get(auc_key) or {}).get("ci_high"),
    }