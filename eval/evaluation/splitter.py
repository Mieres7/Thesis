"""
eval/evaluation/splitter.py

Splits a patient-level DataFrame (one row per uid, after metadata+embeddings
join) into train/val/test partitions, with optional K-fold CV for small
datasets.

Key guarantees:
    - Grouping is done on `uid` (dataset::patient_id), never on raw
      patient_id, to avoid the cross-dataset id-collision bug (BCNB "15"
      vs Histology-3 "15" are different people).
    - Since each row here already IS one patient (slides were pooled
      upstream in embedding_loader.py), grouping by uid is equivalent to a
      plain stratified split -- but we keep the group-based API so this
      still works correctly if pooling strategy ever changes to keep
      multiple rows per patient.
    - Stratification is by `label`, so class proportions are preserved
      across splits as much as the data allows.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from eval.config.datasets import DATASET_EVAL_CONFIG

logger = logging.getLogger(__name__)

SMALL_DATASET_THRESHOLD = 60  # pacientes para activar CV 5-fold (fallback p/datasets nuevos)


# ─────────────────────────────────────────────────────────────────────────────
# Decisión centralizada CV vs. split fijo
#
# Única fuente de verdad: DATASET_EVAL_CONFIG (eval/config/datasets.py).
# Para datasets sin entrada explícita se aplica el fallback por tamaño:
#     n < SMALL_DATASET_THRESHOLD → CV 5-fold (comportamiento conservador
#     por defecto para datasets futuros que nadie configuró a mano).
# NI run_eval.py NI verify_results.py NI ningún otro script deben
# reproducir "if n < umbral": todos deben llamar a estas funciones.
# ─────────────────────────────────────────────────────────────────────────────
def choose_eval_method(dataset_id: str, n_patients: int) -> str:
    """Retorna el método de evaluación ID: ``"cv5"`` o ``"fixed_split"``.

    La config explícita por dataset gana; sólo si el dataset no tiene
    entrada en ``DATASET_EVAL_CONFIG`` se usa el fallback por tamaño.
    """
    cfg = DATASET_EVAL_CONFIG.get(dataset_id)
    if cfg is not None:
        method = cfg.get("method_id")
        if method is not None:
            return method
        return "cv5" if cfg.get("use_cv", False) else "fixed_split"
    return "cv5" if n_patients < SMALL_DATASET_THRESHOLD else "fixed_split"


def should_use_cv(dataset_id: str, n_patients: int) -> bool:
    """True → usar 5-fold CV (OOF combinado cubre el 100% de pacientes)."""
    return choose_eval_method(dataset_id, n_patients) == "cv5"


def eval_method_reason(dataset_id: str, n_patients: int) -> str:
    """Motivo legible de la decisión (para logs y trazabilidad)."""
    cfg = DATASET_EVAL_CONFIG.get(dataset_id)
    if cfg is not None:
        return cfg.get("reason", "configured")
    if n_patients < SMALL_DATASET_THRESHOLD:
        return f"fallback_size_<{SMALL_DATASET_THRESHOLD}"
    return f"fallback_size_>={SMALL_DATASET_THRESHOLD}"


def split_train_val_test(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    group_col: str = "uid",
    label_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """
    Single fixed 70/15/15 split, stratified by label, grouped by uid.

    La decisión de usar este split fijo vs. 5-fold CV la toma
    ``should_use_cv`` (config por dataset en DATASET_EVAL_CONFIG, con
    fallback por SMALL_DATASET_THRESHOLD); este split se usa SOLO cuando
    esa función retorna False.

    Implementation: StratifiedGroupKFold is built for K-fold CV, so we
    repurpose it to carve out test first, then val from the remainder.
    This keeps the "no leakage across groups" guarantee from sklearn
    without needing a bespoke splitting algorithm.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "train/val/test fractions must sum to 1.0"

    groups = df[group_col].values
    labels = df[label_col].values

    n_test_folds = max(2, round(1 / test_frac))
    sgkf_test = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf_test.split(df, labels, groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    val_frac_of_trainval = val_frac / (train_frac + val_frac)
    n_val_folds = max(2, round(1 / val_frac_of_trainval))
    groups_tv = trainval_df[group_col].values
    labels_tv = trainval_df[label_col].values
    sgkf_val = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf_val.split(trainval_df, labels_tv, groups_tv))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    _assert_no_group_leakage(train_df, val_df, test_df, group_col)

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = part[label_col].value_counts(normalize=True).round(3).to_dict()
        logger.info("[%s] n=%d, label distribution=%s", name, len(part), dist)

    return {"train": train_df, "val": val_df, "test": test_df}


def cross_validate_small_dataset(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "uid",
    label_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """
    K-fold cross-validation para datasets chicos (< SMALL_DATASET_THRESHOLD).
    Retorna un dict con train/val/test por fold, donde cada fold usa
    train=(k-1 folds), test=1 fold (sin validation set separado).

    Implementación: StratifiedGroupKFold con grouping por uid.
    """
    groups = df[group_col].values
    labels = df[label_col].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = {}
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(df, labels, groups)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        folds[fold_idx] = {"train": train_df, "val": test_df, "test": test_df}

        logger.info(
            "[fold %d/%d] train=%d, test=%d",
            fold_idx + 1, n_splits, len(train_df), len(test_df),
        )

    return folds


def _assert_no_group_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, group_col: str
) -> None:
    """Hard guarantee: no uid appears in more than one split."""
    train_g = set(train_df[group_col])
    val_g = set(val_df[group_col])
    test_g = set(test_df[group_col])

    overlaps = {
        "train/val": train_g & val_g,
        "train/test": train_g & test_g,
        "val/test": val_g & test_g,
    }
    leaks = {k: v for k, v in overlaps.items() if v}
    if leaks:
        raise RuntimeError(f"Group leakage detected across splits: {leaks}")
