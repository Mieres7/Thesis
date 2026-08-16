"""
src/config/schema.py

Dataclasses that define and validate the YAML configuration for a single
evaluation experiment (one task, e.g. HER2 status, across N datasets).

Kept deliberately small and explicit: no hidden defaults that could silently
change which patients/slides end up in a run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

SubfolderStructure = Literal["case_folders", "flat", "split_folders", "surgen"]


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset entry inside an evaluation config."""
    name: str                        # e.g. "HISTAI_Breast", "BCNB"
    metadata_csv: str                # path to the metadata CSV for this dataset
    id_column: str                   # raw column name holding the patient/case id
    subfolder_structure: SubfolderStructure  # mirrors patch_extractor_opt.py config
    embeddings_dir: str              # output_dir/{dataset}/{model}/ root for this dataset
    stem_filter: Optional[str] = None       # prefix to filter .h5 stems for this dataset (SurGen SR1482/SR386)
    label_column: Optional[str] = None      # task-specific label column (set per task)
    force_label: Optional[str] = None       # if set, overrides label_column with a constant value
    label_map: Optional[dict[str, str]] = None  # harmonize label strings to a canonical vocabulary (per task)
    sex_column: str = "sex"
    age_column: str = "age"

    def __post_init__(self):
        if not self.name:
            raise ValueError("DatasetSpec.name cannot be empty")
        if self.subfolder_structure not in ("case_folders", "flat", "split_folders", "surgen"):
            raise ValueError(
                f"[{self.name}] invalid subfolder_structure: {self.subfolder_structure}"
            )


@dataclass(frozen=True)
class EvalConfig:
    """Top-level config for one evaluation run (one task, one model)."""
    task_name: str                   # e.g. "her2_status", "molecular_subtype", "side", "site"
    model_name: str                  # e.g. "uni2", "virchow2"
    datasets: list[DatasetSpec] = field(default_factory=list)
    output_dir: str = "results"
    seed: int = 42

    def __post_init__(self):
        if not self.datasets:
            raise ValueError(f"[{self.task_name}] no datasets configured")


def load_eval_config(yaml_path: str) -> EvalConfig:
    import yaml
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    datasets = [DatasetSpec(**d) for d in raw["datasets"]]
    return EvalConfig(
        task_name=raw["task_name"],
        model_name=raw["model_name"],
        datasets=datasets,
        output_dir=raw.get("output_dir", "results"),
        seed=raw.get("seed", 42),
    )
