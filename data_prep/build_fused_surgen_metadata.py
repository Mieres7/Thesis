"""
build_fused_surgen_metadata.py

Genera los CSVs de metadata FUSIONADOS para SurGen como un único dataset:

    metadata_clean/colorectal/site/SURGEN_CRC_site.csv   (SR1482 + SR386)
    metadata_clean/colorectal/side/SURGEN_CRC_side.csv   (SR1482 + SR386)

Por qué: SurGen fue curado/armonizado como un solo cohorte (mismo build
de labels site/side). El id de paciente resultante es:

    "{cohorte}_{caso}"  ->  "SR1482_106", "SR386_47"

que es EXACTAMENTE lo que produce embedding_loader.py en modo
subfolder_structure="surgen" (para un archivo SR1482_40X_HE_T106_01.h5 ->
"SR1482_106"). Así los dos SR quedan desambiguados y el join contra los
embeddings es in-ambiguo.

Uso:
    python build_fused_surgen_metadata.py             # escribe los 2 CSV
    python build_fused_surgen_metadata.py --dry-run   # solo muestra conteos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "metadata_clean" / "colorectal" / "site"
SIDE_DIR = ROOT / "metadata_clean" / "colorectal" / "side"


def _fuse(cohorts: list[tuple[str, Path, str]], label_col_out: str) -> pd.DataFrame:
    """Junta los CSVs de una dimensión con id desambiguado por cohorte."""
    frames = []
    for cohort_prefix, csv_path, label_col in cohorts:
        df = pd.read_csv(csv_path)
        caso = df["id"].astype(int).astype(str)
        df["id"] = cohort_prefix + "_" + caso
        df["cohort"] = cohort_prefix
        cols = ["id", "age", "sex", label_col]
        frames.append(df[cols].rename(columns={label_col: label_col_out}))

    fused = pd.concat(frames, ignore_index=True)
    dups = fused["id"].duplicated().sum()
    if dups:
        raise ValueError(f"{dups} ids duplicados tras fusionar")
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    site = _fuse([
        ("SR1482", SITE_DIR / "SURGEN1482_CRC_site.csv", "site"),
        ("SR386", SITE_DIR / "SURGEN386_CRC_site.csv", "site"),
    ], "label")
    side = _fuse([
        ("SR1482", SIDE_DIR / "SURGEN1482_CRC_side.csv", "side"),
        ("SR386", SIDE_DIR / "SURGEN386_CRC_side.csv", "side"),
    ], "label")

    print(f"site: {len(site)} pacientes (SR1482+SR386 fusionados)")
    print(f"side: {len(side)} pacientes (SR1482+SR386 fusionados)")
    print("labels site:", sorted(site["label"].astype(str).unique().tolist()))
    print("labels side:", sorted(side["label"].astype(str).unique().tolist()))
    print("ejemplos ids:", site["id"].head(3).tolist(), "...", side["id"].head(3).tolist())

    if not args.dry_run:
        site.to_csv(SITE_DIR / "SURGEN_CRC_site.csv", index=False)
        side.to_csv(SIDE_DIR / "SURGEN_CRC_side.csv", index=False)
        print("Guardado: SURGEN_CRC_site.csv, SURGEN_CRC_side.csv")


if __name__ == "__main__":
    main()