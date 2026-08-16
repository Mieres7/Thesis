#!/usr/bin/env python3
"""
filter_metadata.py
------------------
Filtra un CSV o XLSX de metadata eliminando filas con valores nulos/missing
en las columnas especificadas en la sección de CONFIGURACIÓN.

Uso:
    python filter_metadata.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# ==============================================================================
#  CONFIGURACIÓN — edita esta sección antes de ejecutar
# ==============================================================================

# Ruta al archivo de entrada (CSV o XLSX)
INPUT_FILE = "./colorectal/surgen1482_final.csv"

# Tarea a ejecutar — define qué columna se filtra.
# Opciones disponibles: "BC1", "BC2", "BC3", "BC4", "CRC1", "CRC2"
# Si prefieres especificar columnas manualmente, deja TASK = None
# y rellena FILTER_COLS.
TASK = "CRC2"

# Columnas a filtrar manualmente (solo se usa si TASK = None)
# Ejemplo: FILTER_COLS = ["ER_status_norm", "PR_status_norm"]
FILTER_COLS = []

# Ruta de salida. Si se deja en None, se genera automáticamente
# en la misma carpeta que el input con el sufijo de la tarea.
OUTPUT_FILE = "./colorectal/surgen1482_final_CRC2.csv"

# Hoja de Excel a leer (solo aplica para .xlsx).
# None = primera hoja. Ejemplo: SHEET = "HISTAI_breast"
SHEET = None

# Separador para archivos CSV. Por defecto coma.
CSV_SEP = ","

# Si True, solo muestra estadísticas sin guardar ningún archivo.
DRY_RUN = False

# ==============================================================================
#  FIN DE CONFIGURACIÓN — no es necesario modificar nada más abajo
# ==============================================================================

MISSING_VALUES = {
    "MISSING", "missing", "nan", "NaN", "NA", "N/A", "n/a",
    "Unknown", "unknown", "UNKNOWN", "", "None", "none"
}

TASK_ALIASES = {
    "BC1":  ["Molecular_subtype_norm"],
    "BC2":  ["HER2_status_norm"],
    "BC3":  ["ER_status_norm"],
    "BC4":  ["PR_status_norm"],
    "CRC1": ["site_group_norm"],
    "CRC2": ["side_norm"],
}


def detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls", ".xlsm"]:
        return "excel"
    elif ext in [".csv", ".tsv", ".txt"]:
        return "csv"
    with open(path, "rb") as f:
        header = f.read(4)
    return "excel" if header[:2] == b"PK" else "csv"


def load_file(path: str) -> pd.DataFrame:
    fmt = detect_format(path)
    if fmt == "excel":
        try:
            df = pd.read_excel(path, sheet_name=SHEET if SHEET else 0, engine="openpyxl")
            sheet_used = SHEET if SHEET else "hoja 1"
            print(f"  Formato: Excel | Hoja: {sheet_used} | {len(df):,} filas | {len(df.columns)} columnas")
            return df
        except Exception as e:
            print(f"[ERROR] No se pudo leer el archivo Excel: {e}")
            print("  Asegúrate de tener instalado: pip install openpyxl")
            sys.exit(1)
    else:
        for enc in ["utf-8", "latin-1", "utf-8-sig"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=CSV_SEP, low_memory=False)
                print(f"  Formato: CSV | Encoding: {enc} | {len(df):,} filas | {len(df.columns)} columnas")
                return df
            except UnicodeDecodeError:
                continue
        print(f"[ERROR] No se pudo leer {path}")
        sys.exit(1)


def is_missing(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() in MISSING_VALUES


def filter_dataframe(df: pd.DataFrame, filter_cols: list) -> tuple:
    stats = {"original": len(df), "removed_per_col": {}, "final": 0, "total_removed": 0}

    missing_cols = [c for c in filter_cols if c not in df.columns]
    if missing_cols:
        print(f"\n[ERROR] Columnas no encontradas: {missing_cols}")
        print(f"  Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    mask_keep = pd.Series([True] * len(df), index=df.index)
    for col in filter_cols:
        missing_mask = df[col].apply(is_missing)
        stats["removed_per_col"][col] = int(missing_mask.sum())
        mask_keep &= ~missing_mask

    df_filtered = df[mask_keep].reset_index(drop=True)
    stats["final"] = len(df_filtered)
    stats["total_removed"] = stats["original"] - stats["final"]
    return df_filtered, stats


def build_output_path(filter_cols: list, task: str, out_fmt: str) -> str:
    stem = Path(INPUT_FILE).stem
    parent = Path(INPUT_FILE).parent
    ext = ".xlsx" if out_fmt == "excel" else ".csv"
    suffix = f"_filtered_{task}" if task else "_filtered_" + "_".join(
        c.replace("_norm", "").replace("_status", "") for c in filter_cols
    )
    return str(parent / f"{stem}{suffix}{ext}")


def save_file(df: pd.DataFrame, output_path: str):
    ext = Path(output_path).suffix.lower()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if ext in [".xlsx", ".xls"]:
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"\nGuardado como Excel en: {output_path}")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nGuardado como CSV en: {output_path}")


def print_stats(stats: dict):
    print(f"\n{'─'*50}")
    print(f"  Filas originales : {stats['original']:,}")
    for col, n in stats["removed_per_col"].items():
        pct = n / stats["original"] * 100 if stats["original"] > 0 else 0
        print(f"  MISSING en '{col}': {n:,} ({pct:.1f}%)")
    print(f"  Total eliminadas : {stats['total_removed']:,}")
    print(f"  Filas finales    : {stats['final']:,} ({stats['final']/stats['original']*100:.1f}% retenido)")
    print(f"{'─'*50}")


def main():
    # Resolver columnas a filtrar
    if TASK:
        if TASK not in TASK_ALIASES:
            print(f"[ERROR] Tarea '{TASK}' no reconocida. Opciones: {list(TASK_ALIASES.keys())}")
            sys.exit(1)
        filter_cols = TASK_ALIASES[TASK]
        print(f"\nTarea: {TASK} → filtrando por: {filter_cols}")
    elif FILTER_COLS:
        filter_cols = FILTER_COLS
        print(f"\nFiltrando por columnas: {filter_cols}")
    else:
        print("[ERROR] Debes definir TASK o FILTER_COLS en la sección de configuración.")
        sys.exit(1)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontró el archivo: {INPUT_FILE}")
        sys.exit(1)

    print(f"\nCargando: {INPUT_FILE}")
    df = load_file(INPUT_FILE)

    df_filtered, stats = filter_dataframe(df, filter_cols)
    print_stats(stats)

    if DRY_RUN:
        print("\n[DRY RUN] No se guardó ningún archivo.")
        return

    out_fmt = detect_format(INPUT_FILE)
    output_path = OUTPUT_FILE if OUTPUT_FILE else build_output_path(filter_cols, TASK, out_fmt)
    save_file(df_filtered, output_path)


if __name__ == "__main__":
    main()