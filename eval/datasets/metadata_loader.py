"""
eval/datasets/metadata_loader.py

Loads and normalizes per-dataset metadata CSVs into a common schema:
    patient_id, age, sex, label, dataset

IMPORTANT: patient_id is only unique WITHIN a dataset. Numeric-id datasets
(BCNB, Histology-3, SurGen) can and do collide with each other (e.g. both
have a patient "15"). The canonical unique key everywhere downstream is the
pair (dataset, patient_id), exposed as a precomputed `uid` column
(f"{dataset}::{patient_id}") to avoid repeating the concat logic everywhere.

Handles the two id conventions seen across datasets:
    - HISTAI-style: "histai/HISTAI-breast/case_1551"  -> "case_1551"
    - Direct id:    "42"                              -> "42"

Missing labels are dropped per-task (each task only trains/evaluates on the
patients that actually have that label), exactly matching how the raw CSVs
were already curated.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from eval.config.schema import DatasetSpec

logger = logging.getLogger(__name__)


def _normalize_id(raw_id: str) -> str:
    """
    Normalize a raw id value into the canonical patient_id used everywhere
    downstream (embeddings, splits, results).

    HISTAI-style values are paths ("histai/HISTAI-breast/case_1551");
    everything else (BCNB, SurGen, Histology-3) is already a direct id.
    """
    raw_id = str(raw_id).strip()
    if "/" in raw_id:
        return raw_id.rstrip("/").split("/")[-1]
    return raw_id


def make_uid(dataset: str, patient_id: str) -> str:
    """
    Canonical globally-unique patient key. patient_id alone is NOT safe to
    join/group on across datasets -- numeric-id datasets (BCNB, Histology-3,
    SurGen) collide (e.g. patient "15" exists in both BCNB and Histology-3
    as different people).
    """
    return f"{dataset}::{patient_id}"


def load_dataset_metadata(spec: DatasetSpec) -> pd.DataFrame:
    """
    Read one dataset's metadata CSV and return a normalized DataFrame with
    columns: uid, patient_id, age, sex, label, dataset.

    Rows with a missing label for this task are dropped here -- this mirrors
    how the CSVs were curated (filtered per-task upstream).
    """
    csv_path = Path(spec.metadata_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"[{spec.name}] metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = [spec.id_column]
    if spec.label_column is not None:
        required.append(spec.label_column)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"[{spec.name}] expected columns {missing_cols} not found in "
            f"{csv_path.name}. Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["patient_id"] = df[spec.id_column].map(_normalize_id)
    out["dataset"] = spec.name
    out["uid"] = [make_uid(spec.name, pid) for pid in out["patient_id"]]

    out["age"] = df[spec.age_column] if spec.age_column in df.columns else pd.NA
    out["sex"] = df[spec.sex_column] if spec.sex_column in df.columns else pd.NA

    if spec.force_label is not None:
        out["label"] = spec.force_label
        logger.info("[%s] forced label='%s' for all %d rows", spec.name, spec.force_label, len(out))
    elif spec.label_column is not None:
        out["label"] = df[spec.label_column]
        n_before = len(out)
        out = out.dropna(subset=["label"])
        n_after = len(out)
        if n_before != n_after:
            logger.info(
                "[%s] dropped %d/%d rows with missing label (%s)",
                spec.name, n_before - n_after, n_before, spec.label_column,
            )
        if spec.label_map:
            raw_labels = out["label"].unique().tolist()
            unmapped = sorted(set(raw_labels) - set(spec.label_map.keys()))
            if unmapped:
                logger.warning(
                    "[%s] labels sin mapeo en label_map: %s", spec.name, unmapped,
                )
            out["label"] = out["label"].map(lambda v: spec.label_map.get(v, v))
    else:
        out["label"] = pd.NA

    n_dupes = out["uid"].duplicated().sum()
    if n_dupes:
        raise ValueError(
            f"[{spec.name}] {n_dupes} duplicated uid values within this "
            f"dataset's own metadata -- check id_column mapping, this "
            f"should never happen within a single dataset."
        )

    logger.info("[%s] loaded %d labeled patients", spec.name, len(out))
    return out[["uid", "patient_id", "dataset", "age", "sex", "label"]].reset_index(drop=True)


def load_all_metadata(specs: list[DatasetSpec]) -> pd.DataFrame:
    """
    Load and concatenate metadata for every dataset in a task config.
    Safe against cross-dataset id collisions because `uid` (dataset::id)
    is the join/group key, not `patient_id` alone.
    """
    frames = [load_dataset_metadata(s) for s in specs]
    combined = pd.concat(frames, ignore_index=True)

    n_dupes = combined["uid"].duplicated().sum()
    assert n_dupes == 0, (
        f"Found {n_dupes} duplicated uid across combined metadata -- "
        f"this should be impossible since uid embeds the dataset name."
    )

    logger.info(
        "Loaded metadata for %d datasets, %d total labeled patients "
        "(patient_id collisions across datasets are safely disambiguated by uid)",
        len(specs), len(combined),
    )
    return combined
