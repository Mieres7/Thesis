# """
# src/datasets/embedding_loader.py

# Reads per-slide embedding .h5 files produced by embed_from_patches.py:

#     output_dir/{dataset}/{model}/{slide_stem}.h5
#         embeddings: (N, D) float32
#         coords:     (N, 2) int32

# and pools them into one feature vector per patient:

#     patch embeddings --mean--> slide embedding --mean--> patient embedding

# Patient identity is resolved from the .h5 filename using the same
# subfolder_structure convention as patch_extractor_opt.py:

# - "case_folders" (HISTAI): filename is "{patient_id}_{slide_suffix}.h5"
#       e.g. "case_1551_slideA.h5" -> patient_id "case_1551"
#     - "flat" (BCNB, Histology-3): filename IS the patient_id
#       e.g. "42.h5" -> patient_id "42"
#     - "surgen" (SurGen): the case id is embedded in the slide name
#       e.g. "SR1482_40X_HE_T106_01.h5" -> patient_id "106"

# The result is joined against metadata via `uid` (dataset::patient_id), the
# same canonical key used in metadata_loader.py, so numeric-id collections
# across datasets (BCNB "15" vs Histology-3 "15") are never mixed.
# """
# from __future__ import annotations

# import logging
# import re
# from pathlib import Path

# import h5py
# import numpy as np
# import pandas as pd

# from eval.config.schema import DatasetSpec
# from eval.datasets.metadata_loader import make_uid

# logger = logging.getLogger(__name__)

# _CASE_ID_RE = re.compile(r"(case_\d+)")
# _SURGEN_ID_RE = re.compile(r"_T0*(\d+)_")


# def _extract_patient_id_from_stem(h5_stem: str, subfolder_structure: str) -> str:
#     """Resolve patient_id from a .h5 filename stem given the dataset's layout."""
#     if subfolder_structure == "case_folders":
#         match = _CASE_ID_RE.match(h5_stem)
#         if not match:
#             raise ValueError(
#                 f"Expected 'case_XXXX' prefix in filename '{h5_stem}.h5' "
#                 f"for subfolder_structure='case_folders'"
#             )
#         return match.group(1)
#     elif subfolder_structure in ("flat", "split_folders"):
#         # the whole stem IS the patient id (BCNB: "123", Histology-3: "42", ...)
#         return h5_stem
#     elif subfolder_structure == "surgen":
#         # SurGen: case id embedded in slide name. e.g. "SR1482_40X_HE_T106_01"
#         # -> "SR1482_106" (prefijo de cohorte + caso). El prefijo desambigua
#         # SR1482 vs SR386, cuyos ids de caso se superponen entre sí.
#         match = _SURGEN_ID_RE.search(h5_stem)
#         if not match:
#             raise ValueError(
#                 f"Expected '_T<case_id>_' in filename '{h5_stem}.h5' "
#                 f"for subfolder_structure='surgen'"
#             )
#         cohort = h5_stem.split("_")[0]
#         case_id = match.group(1).lstrip("0") or "0"
#         return f"{cohort}_{case_id}"
#     raise ValueError(f"Unsupported subfolder_structure: {subfolder_structure}")


# def _read_slide_embedding(h5_path: Path, pooling: str = "mean") -> np.ndarray:
#     """Read one slide's patch embeddings and pool them into a single vector."""
#     with h5py.File(h5_path, "r") as f:
#         embeds = f["embeddings"][:]  # (N, D) float32
#     if embeds.shape[0] == 0:
#         raise ValueError(f"{h5_path} has zero patches")
#     if pooling == "mean":
#         return embeds.mean(axis=0)
#     raise ValueError(f"Unsupported pooling: {pooling}")


# def load_dataset_embeddings(
#     spec: DatasetSpec,
#     model_name: str,
#     pooling: str = "mean",
# ) -> pd.DataFrame:
#     """
#     Discover every slide .h5 for (dataset, model), pool patches -> slide
#     vectors, group slides -> patient vectors (mean over slides of the same
#     patient), and return a DataFrame indexed by uid with columns:
#         uid, dataset, patient_id, n_slides, embedding (object: np.ndarray)

#     Returns empty DataFrame if embeddings dir doesn't exist or has no .h5 files
#     (logs a warning instead of raising).
#     """
#     model_dir = Path(spec.embeddings_dir) / model_name
#     if not model_dir.exists():
#         logger.warning(
#             "[%s/%s] embeddings dir not found: %s -- skipping",
#             spec.name, model_name, model_dir,
#         )
#         return pd.DataFrame(columns=["uid", "dataset", "patient_id", "n_slides", "embedding"])

#     h5_files = sorted(model_dir.glob("*.h5"))
#     if spec.stem_filter:
#         h5_files = [f for f in h5_files if f.stem.startswith(spec.stem_filter)]
#         if not h5_files:
#             logger.warning(
#                 "[%s/%s] stem_filter='%s' no match in %s -- skipping",
#                 spec.name, model_name, spec.stem_filter, model_dir,
#             )
#             return pd.DataFrame(columns=["uid", "dataset", "patient_id", "n_slides", "embedding"])
#     if not h5_files:
#         logger.warning(
#             "[%s/%s] no .h5 files found in %s -- skipping",
#             spec.name, model_name, model_dir,
#         )
#         return pd.DataFrame(columns=["uid", "dataset", "patient_id", "n_slides", "embedding"])

#     slide_vectors: dict[str, list[np.ndarray]] = {}
#     errors = 0
#     for h5_path in h5_files:
#         try:
#             patient_id = _extract_patient_id_from_stem(
#                 h5_path.stem, spec.subfolder_structure
#             )
#             vec = _read_slide_embedding(h5_path, pooling=pooling)
#             slide_vectors.setdefault(patient_id, []).append(vec)
#         except Exception as e:
#             logger.error("[%s/%s] failed on %s: %s", spec.name, model_name, h5_path.name, e)
#             errors += 1

#     if errors:
#         logger.warning(
#             "[%s/%s] %d/%d slide files failed to load",
#             spec.name, model_name, errors, len(h5_files)
#         )

#     rows = []
#     for patient_id, vecs in slide_vectors.items():
#         patient_embedding = np.mean(np.stack(vecs, axis=0), axis=0)
#         rows.append({
#             "uid": make_uid(spec.name, patient_id),
#             "dataset": spec.name,
#             "patient_id": patient_id,
#             "n_slides": len(vecs),
#             "embedding": patient_embedding,
#         })

#     df = pd.DataFrame(rows)
#     logger.info(
#         "[%s/%s] pooled %d slides -> %d patients (mean %.1f slides/patient)",
#         spec.name, model_name, len(h5_files), len(df),
#         df["n_slides"].mean() if len(df) else 0.0,
#     )
#     return df


# def load_all_embeddings(
#     specs: list[DatasetSpec],
#     model_name: str,
#     pooling: str = "mean",
# ) -> pd.DataFrame:
#     """Load and concatenate patient-level embeddings across every dataset."""
#     frames = [load_dataset_embeddings(s, model_name, pooling=pooling) for s in specs]
    
#     # Filter out empty DataFrames (datasets with missing embeddings)
#     frames = [f for f in frames if len(f) > 0]
    
#     if not frames:
#         logger.error(
#             "[%s] no datasets have embeddings available", model_name
#         )
#         return pd.DataFrame(columns=["uid", "dataset", "patient_id", "n_slides", "embedding"])
    
#     combined = pd.concat(frames, ignore_index=True)

#     n_dupes = combined["uid"].duplicated().sum()
#     assert n_dupes == 0, f"Found {n_dupes} duplicated uid in embeddings -- should be impossible"

#     logger.info(
#         "[%s] loaded embeddings for %d datasets, %d total patients",
#         model_name, len(specs), len(combined),
#     )
#     return combined


# def join_metadata_and_embeddings(
#     metadata_df: pd.DataFrame,
#     embeddings_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Inner join on uid. Logs (and does not silently swallow) any patients
#     present in metadata but missing embeddings, or vice versa -- this is
#     the single most important sanity check in the whole pipeline.
#     """
#     meta_uids = set(metadata_df["uid"])
#     emb_uids = set(embeddings_df["uid"])

#     only_in_meta = meta_uids - emb_uids
#     only_in_emb = emb_uids - meta_uids

#     if only_in_meta:
#         logger.warning(
#             "%d patients have metadata/label but NO embeddings (excluded): %s",
#             len(only_in_meta),
#             list(only_in_meta)[:5],
#         )
#     if only_in_emb:
#         logger.warning(
#             "%d patients have embeddings but NO metadata/label (excluded): %s",
#             len(only_in_emb),
#             list(only_in_emb)[:5],
#         )

#     merged = metadata_df.merge(
#         embeddings_df[["uid", "n_slides", "embedding"]], on="uid", how="inner"
#     )
#     logger.info(
#         "Joined metadata + embeddings: %d patients retained "
#         "(%d from metadata, %d from embeddings)",
#         len(merged), len(metadata_df), len(embeddings_df),
#     )
#     return merged


"""
eval/datasets/embedding_loader.py

Optimized version: caches pooled patient-level embeddings in RAM during one
run. A dataset/model pair is read from .h5 files only once; later tasks that
reuse it receive the cached patient-level DataFrame.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eval.config.schema import DatasetSpec
from eval.datasets.metadata_loader import make_uid

logger = logging.getLogger(__name__)

_CASE_ID_RE = re.compile(r"(case_\d+)")
_SURGEN_ID_RE = re.compile(r"_T0*(\d+)_")

# Se guardan vectores ya pooled a nivel paciente, no los embeddings por patch.
# Por tanto, el uso de RAM es mucho menor que volver a mantener los H5 completos.
_DATASET_EMBEDDING_CACHE: dict[tuple[str, str, str, str, str], pd.DataFrame] = {}


def _cache_key(spec: DatasetSpec, model_name: str, pooling: str) -> tuple[str, str, str, str, str]:
    return (
        str(Path(spec.embeddings_dir).resolve()),
        spec.name,
        model_name,
        spec.stem_filter or "",
        pooling,
    )


def clear_embedding_cache() -> None:
    """Libera los embeddings pooled cacheados en memoria."""
    n_items = len(_DATASET_EMBEDDING_CACHE)
    _DATASET_EMBEDDING_CACHE.clear()
    logger.info("Cleared embedding cache (%d dataset/model entries)", n_items)


def embedding_cache_info() -> dict[str, int]:
    """Devuelve un resumen simple para logging o depuración."""
    return {
        "entries": len(_DATASET_EMBEDDING_CACHE),
        "patients": sum(len(df) for df in _DATASET_EMBEDDING_CACHE.values()),
    }


def _extract_patient_id_from_stem(h5_stem: str, subfolder_structure: str) -> str:
    """Resolve patient_id from a .h5 filename stem given the dataset layout."""
    if subfolder_structure == "case_folders":
        match = _CASE_ID_RE.match(h5_stem)
        if not match:
            raise ValueError(
                f"Expected 'case_XXXX' prefix in filename '{h5_stem}.h5' "
                "for subfolder_structure='case_folders'"
            )
        return match.group(1)

    if subfolder_structure in ("flat", "split_folders"):
        return h5_stem

    if subfolder_structure == "surgen":
        match = _SURGEN_ID_RE.search(h5_stem)
        if not match:
            raise ValueError(
                f"Expected '_T_' in filename '{h5_stem}.h5' "
                "for subfolder_structure='surgen'"
            )
        cohort = h5_stem.split("_")[0]
        case_id = match.group(1).lstrip("0") or "0"
        return f"{cohort}_{case_id}"

    raise ValueError(f"Unsupported subfolder_structure: {subfolder_structure}")


def _read_slide_embedding(h5_path: Path, pooling: str = "mean") -> np.ndarray:
    """Read one slide's patch embeddings and pool them into one vector."""
    with h5py.File(h5_path, "r") as f:
        embeds = f["embeddings"][:]

    if embeds.shape[0] == 0:
        raise ValueError(f"{h5_path} has zero patches")
    if pooling == "mean":
        return embeds.mean(axis=0)
    raise ValueError(f"Unsupported pooling: {pooling}")


def load_dataset_embeddings(
    spec: DatasetSpec,
    model_name: str,
    pooling: str = "mean",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Pool all slides for one dataset/model to patient-level vectors.

    With use_cache=True (default), an already pooled dataset/model is reused
    within the current Python process. The source .h5 files are not reopened.
    """
    key = _cache_key(spec, model_name, pooling)
    if use_cache and key in _DATASET_EMBEDDING_CACHE:
        cached = _DATASET_EMBEDDING_CACHE[key]
        logger.info(
            "[%s/%s] CACHE HIT: reusing %d patient embeddings",
            spec.name,
            model_name,
            len(cached),
        )
        return cached

    model_dir = Path(spec.embeddings_dir) / model_name
    empty = pd.DataFrame(columns=["uid", "dataset", "patient_id", "n_slides", "embedding"])

    if not model_dir.exists():
        logger.warning("[%s/%s] embeddings dir not found: %s -- skipping", spec.name, model_name, model_dir)
        return empty

    h5_files = sorted(model_dir.glob("*.h5"))
    if spec.stem_filter:
        h5_files = [path for path in h5_files if path.stem.startswith(spec.stem_filter)]
    if not h5_files:
        logger.warning("[%s/%s] no matching .h5 files in %s -- skipping", spec.name, model_name, model_dir)
        return empty

    slide_vectors: dict[str, list[np.ndarray]] = {}
    errors = 0
    for h5_path in h5_files:
        try:
            patient_id = _extract_patient_id_from_stem(h5_path.stem, spec.subfolder_structure)
            slide_vectors.setdefault(patient_id, []).append(_read_slide_embedding(h5_path, pooling))
        except Exception as error:
            logger.error("[%s/%s] failed on %s: %s", spec.name, model_name, h5_path.name, error)
            errors += 1

    if errors:
        logger.warning("[%s/%s] %d/%d slide files failed to load", spec.name, model_name, errors, len(h5_files))

    rows = [
        {
            "uid": make_uid(spec.name, patient_id),
            "dataset": spec.name,
            "patient_id": patient_id,
            "n_slides": len(vectors),
            "embedding": np.mean(np.stack(vectors, axis=0), axis=0),
        }
        for patient_id, vectors in slide_vectors.items()
    ]
    df = pd.DataFrame(rows, columns=["uid", "dataset", "patient_id", "n_slides", "embedding"])

    if use_cache:
        _DATASET_EMBEDDING_CACHE[key] = df

    logger.info(
        "[%s/%s] pooled %d slides -> %d patients (mean %.1f slides/patient)%s",
        spec.name,
        model_name,
        len(h5_files),
        len(df),
        df["n_slides"].mean() if len(df) else 0.0,
        "; cached" if use_cache else "",
    )
    return df


def load_all_embeddings(
    specs: list[DatasetSpec],
    model_name: str,
    pooling: str = "mean",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load patient embeddings across task datasets, reusing RAM cache when possible."""
    frames = [
        load_dataset_embeddings(spec, model_name, pooling=pooling, use_cache=use_cache)
        for spec in specs
    ]
    frames = [frame for frame in frames if len(frame) > 0]

    if not frames:
        logger.error("[%s] no datasets have embeddings available", model_name)
        return pd.DataFrame(columns=["uid", "dataset", "patient_id", "n_slides", "embedding"])

    combined = pd.concat(frames, ignore_index=True)
    n_dupes = combined["uid"].duplicated().sum()
    assert n_dupes == 0, f"Found {n_dupes} duplicated uid in embeddings -- should be impossible"

    logger.info("[%s] loaded embeddings for %d datasets, %d total patients", model_name, len(specs), len(combined))
    return combined


def join_metadata_and_embeddings(metadata_df: pd.DataFrame, embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """Inner join on uid, retaining mismatch diagnostics."""
    meta_uids = pd.Index(metadata_df["uid"])
    emb_uids = pd.Index(embeddings_df["uid"])
    only_in_meta = meta_uids.difference(emb_uids)
    only_in_emb = emb_uids.difference(meta_uids)

    if len(only_in_meta):
        logger.warning(
            "%d patients have metadata/label but NO embeddings (excluded): %s",
            len(only_in_meta),
            only_in_meta[:5].tolist(),
        )
    if len(only_in_emb):
        logger.warning(
            "%d patients have embeddings but NO metadata/label (excluded): %s",
            len(only_in_emb),
            only_in_emb[:5].tolist(),
        )

    merged = metadata_df.merge(
        embeddings_df[["uid", "n_slides", "embedding"]],
        on="uid",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    logger.info(
        "Joined metadata + embeddings: %d patients retained (%d metadata, %d embeddings)",
        len(merged),
        len(metadata_df),
        len(embeddings_df),
    )
    return merged