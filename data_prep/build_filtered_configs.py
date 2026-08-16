#!/usr/bin/env python3
"""
build_filtered_configs.py
=========================
Lee los CSVs de metadata_clean/ y genera configs YAML con select filtrado
para que patch_extractor_opt.py y embed_from_patches_opt.py solo procesen
las slides que realmente necesitás.

Uso:
    python build_filtered_configs.py
    python build_filtered_configs.py --dry-run
    python build_filtered_configs.py --task breast
    python build_filtered_configs.py --task colorectal

Genera:
    config_files/patches_<dataset>_filtered.yaml
    config_files/embeddings_<dataset>_filtered.yaml
"""

import os
import re
import csv
import yaml
import argparse
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# RUTAS CONFIGURABLES — Editar aquí al pasar al servidor
# ═══════════════════════════════════════════════════════════════════════════════

PATHS = {
    "metadata_clean": "./metadata_clean",

    # --- Breast ---
    "bcnb_slides": "/home/DIINF/datasets_iacis/2026_BCNB/WSIs",
    "histai_breast_slides": "/home/shared_data/HISTAI/HISTAI_Breast",
    "histology_slides": "/home/DIINF/mieres/PKG - HistologyHSI-BC-Recurrence/01_01_Histological_Images",

    # --- Colorectal ---
    "histai_b1_slides": "/home/shared_data/HISTAI/HISTAI_Colorectal_b1",
    "histai_b2_slides": "/home/DIINF/datasets_iacis/2026_HISTAI/HISTAI_colorectal_b2",
    "surgen_386_slides": "/home/shared_data/SurGen/SR386_WSIs",
    "surgen_1482_slides": "/home/shared_data/SurGen/SR1482_WSIs",
    "surgen_parent": "/home/shared_data/SurGen",

    # --- Models & output ---
    "models_dir": "/home/DIINF/vmieres/tesis/models",
    "patches_base": "/home/DIINF/datasets_iacis/2026_output/patches",
    "embeddings_base": "/home/DIINF/datasets_iacis/2026_output/embeddings",
}

# ═══════════════════════════════════════════════════════════════════════════════
# DEFINICIÓN DE TAREAS Y DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

TASKS = {
    "breast":{
        "datasets":[]
    },
    "colorectal": {
        "datasets": [
            {
                "name": "SURGEN",
                "csv_patterns": [
                    "/home/vmieres/tesis/metadata_clean/colorectal/site/SURGEN386_CRC_site.csv",
                    "/home/vmieres/tesis/metadata_clean/colorectal/side/SURGEN386_CRC_side.csv",
                    "/home/vmieres/tesis/metadata_clean/colorectal/site/SURGEN1482_CRC_site.csv",
                    "/home/vmieres/tesis/metadata_clean/colorectal/side/SURGEN1482_CRC_side.csv",
                ],
                "id_column": "id",
                "slides_paths": [
                    {"path_key": "surgen_386_slides", "prefix": "SR386"},
                    {"path_key": "surgen_1482_slides", "prefix": "SR1482"},
                ],
                "slides_path_key": "surgen_parent",
                "format": "czi",
                "subfolder_structure": "split_folders",
                "id_type": "surgen_combined",
                "native_magnification": 40,
            },
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATES YAML (se llenan con select filtrado)
# ═══════════════════════════════════════════════════════════════════════════════

PATCHES_TEMPLATE = {
    "log_level": "INFO",
    "log_file": None,  # se llena después
    "compute": {
        "gpu_ids": [0, 1],
        "num_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": True,
    },
    "extraction": {
        "patch_size": 224,
        "magnification": 20.0,
        "native_magnification": 40,
        "overlap": 0.0,
        "padding": True,
    },
    "filtering": {
        "enabled": True,
        "tissue_threshold": 0.5,
        "saturation_filter": True,
        "saturation_threshold": 0.05,
    },
    "output": {
        "base_dir": None,  # se llena
        "save_patches": True,
        "patches_format": "hdf5",
        "hdf5_compression": "lzf",
        "save_embeddings": False,
        "embedding_model": "uni2",
        "models_dir": None,  # se llena
        "embedding_batch_size": 64,
        "skip_existing": False,
    },
    "datasets": [],  # se llena
}

EMBEDDINGS_TEMPLATE = {
    "log_level": "INFO",
    "log_file": None,
    "patches_dir": None,
    "output_dir": None,
    "models_dir": None,
    "models": ["uni2", "virchow2", "phikon_v2", "prov_gigapath"],
    "datasets": [],  # solo nombres
    "batch_size": 256,
    "skip_existing": True,
    "hdf5_compression": "lzf",
    "use_amp": True,
    "compute": {
        "gpu_ids": [0, 1],
        "num_workers": 4,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ═══════════════════════════════════════════════════════════════════════════════

def read_csv_ids(csv_path: str, id_column: str) -> set:
    """Lee un CSV y retorna el set de valores únicos de la columna ID.
    Si la columna exacta no existe, busca alternativas comunes (ej: 'is' por typo de 'id')."""
    ids = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        # Buscar la columna real (maneja typo 'is' → 'id')
        real_col = id_column
        if real_col not in headers:
            alternatives = {"id": ["is"], "is": ["id"]}
            for alt in alternatives.get(id_column, []):
                if alt in headers:
                    real_col = alt
                    print(f"  INFO: Columna '{id_column}' no encontrada, usando '{alt}'")
                    break
        for row in reader:
            val = row.get(real_col, "").strip()
            if val:
                ids.add(val)
    return ids


def extract_case_dir_name(case_mapping: str) -> str:
    """De 'histai/HISTAI-colorectal-b1/case_414' extrae 'case_414'."""
    return Path(case_mapping).name


def extract_numeric_stem(id_val: str) -> str:
    """De un ID numérico (ej: '15'), retorna el string tal cual para matching."""
    return str(id_val).strip()


def scan_surgen_files(slides_dir: str, prefix: str) -> dict:
    """
    Escanea directorio SURGEN y retorna {id_num: [filename_stems]}.
    Ej: {'44': ['SR1482_40X_HE_T044_01', 'SR1482_40X_HE_T044_02'], ...}
    """
    slides_path = Path(slides_dir)
    if not slides_path.exists():
        print(f"  WARNING: SURGEN dir no existe: {slides_dir}")
        return {}

    case_to_files = defaultdict(list)
    for f in slides_path.iterdir():
        if f.suffix.lower() == ".czi" and f.stem.startswith(prefix):
            # Buscar patrón T{digits} en el nombre
            # Ej: SR1482_40X_HE_T044_01 → T044 → 44
            # Ej: SR386_40X_HE_T03_01 → T03 → 3
            match = re.search(r'_T0*(\d+)_', f.stem)
            if match:
                case_id = match.group(1)
                case_to_files[case_id].append(f.stem)
            else:
                print(f"  WARNING: No se pudo extraer case ID de: {f.name}")

    return dict(case_to_files)


def resolve_select_ids(dataset_cfg: dict, all_ids: set, paths: dict) -> list:
    """
    Dado un dataset config y los IDs únicos del CSV, retorna la lista de
    stems/directorios para el campo select del YAML.
    """
    id_type = dataset_cfg["id_type"]
    slides_dir = paths.get(dataset_cfg["slides_path_key"], "")

    if id_type == "case_dir":
        # HISTAI: case_mapping → nombre de directorio
        return sorted([extract_case_dir_name(cid) for cid in all_ids])

    elif id_type == "numeric_stem":
        # BCNB / Histology: ID numérico = stem del archivo
        return sorted([extract_numeric_stem(cid) for cid in all_ids])

    elif id_type == "surgen_pattern":
        # SURGEN (individual): mapear IDs del CSV a archivos reales en disco
        prefix = dataset_cfg["surgen_prefix"]
        case_to_files = scan_surgen_files(slides_dir, prefix)

        matched_files = []
        unmatched_ids = []
        for cid in sorted(all_ids, key=lambda x: int(x) if x.isdigit() else 0):
            cid_str = str(cid).strip()
            if cid_str in case_to_files:
                matched_files.extend(case_to_files[cid_str])
            else:
                unmatched_ids.append(cid_str)

        if unmatched_ids:
            print(f"  WARNING: {dataset_cfg['name']}: {len(unmatched_ids)} IDs sin archivos en disco: "
                  f"{unmatched_ids[:10]}{'...' if len(unmatched_ids) > 10 else ''}")

        return sorted(matched_files)

    elif id_type == "surgen_combined":
        # SURGEN combinado: escanea múltiples directorios con distintos prefijos
        matched_files = []
        all_unmatched = set(all_ids)

        for slide_cfg in dataset_cfg["slides_paths"]:
            slides_dir = paths.get(slide_cfg["path_key"], "")
            prefix = slide_cfg["prefix"]
            case_to_files = scan_surgen_files(slides_dir, prefix)

            for cid in sorted(all_ids, key=lambda x: int(x) if x.isdigit() else 0):
                cid_str = str(cid).strip()
                if cid_str in case_to_files:
                    matched_files.extend(case_to_files[cid_str])
                    all_unmatched.discard(cid_str)

        if all_unmatched:
            print(f"  WARNING: {dataset_cfg['name']}: {len(all_unmatched)} IDs sin archivos en disco: "
                  f"{sorted(all_unmatched)[:10]}{'...' if len(all_unmatched) > 10 else ''}")

        return sorted(set(matched_files))

    else:
        raise ValueError(f"ID type desconocido: {id_type}")


def generate_patches_yaml(dataset_entry: dict, select_list: list,
                           task_name: str, paths: dict) -> dict:
    """Genera el dict YAML para un dataset de patches."""
    cfg = yaml.safe_load(yaml.dump(PATCHES_TEMPLATE))  # deep copy

    slug = dataset_entry["name"].lower().replace(" ", "_")
    cfg["log_file"] = f"patches_logs/patches_{slug}_filtered.log"
    cfg["output"]["base_dir"] = str(Path(paths["patches_base"]) / dataset_entry["name"])
    cfg["output"]["models_dir"] = paths["models_dir"]

    # Ajustar tissue_threshold según dataset
    if "breast" in task_name and "bcnb" in slug:
        cfg["filtering"]["tissue_threshold"] = 0.5
        cfg["filtering"]["saturation_threshold"] = 0.03
    elif "breast" in task_name:
        cfg["filtering"]["tissue_threshold"] = 0.8
        cfg["filtering"]["saturation_threshold"] = 0.03
    else:
        cfg["filtering"]["tissue_threshold"] = 0.5
        cfg["filtering"]["saturation_threshold"] = 0.05

    # Configurar native_magnification si el dataset lo define
    if "native_magnification" in dataset_entry:
        cfg["extraction"]["native_magnification"] = dataset_entry["native_magnification"]

    ds_entry = {
        "name": dataset_entry["name"],
        "enabled": True,
        "path": paths.get(dataset_entry["slides_path_key"], ""),
        "format": dataset_entry["format"],
        "subfolder_structure": dataset_entry["subfolder_structure"],
        "select": select_list,
    }
    cfg["datasets"] = [ds_entry]

    return cfg


def generate_embeddings_yaml(dataset_entry: dict, task_name: str, paths: dict) -> dict:
    """Genera el dict YAML para embeddings de un dataset."""
    cfg = yaml.safe_load(yaml.dump(EMBEDDINGS_TEMPLATE))  # deep copy

    slug = dataset_entry["name"].lower().replace(" ", "_")
    cfg["log_file"] = f"embeddings_logs/embeddings_{slug}_filtered.log"
    cfg["patches_dir"] = str(Path(paths["patches_base"]) / dataset_entry["name"])
    cfg["output_dir"] = str(Path(paths["embeddings_base"]) / dataset_entry["name"])
    cfg["models_dir"] = paths["models_dir"]
    cfg["datasets"] = [dataset_entry["name"]]

    return cfg


def count_slides_in_select(select_list: list, slides_dir: str,
                            subfolder_structure: str, fmt: str) -> int:
    """
    Cuenta el número real de archivos de slide dentro de los directorios/selecciones.
    Para case_folders: cuenta archivos dentro de cada subcarpeta seleccionada.
    Para flat: el select ya son los stems de los archivos (1:1).
    """
    base = Path(slides_dir)
    if not base.exists():
        return -1  # no se puede contar

    ext_map = {
        "mrxs": [".mrxs"], "jpg": [".jpg", ".jpeg"], "tiff": [".tiff", ".tif"],
        "czi": [".czi"], "svs": [".svs"], "ndpi": [".ndpi"],
    }
    extensions = ext_map.get(fmt, [f".{fmt}"])

    if subfolder_structure == "case_folders":
        count = 0
        for case_name in select_list:
            case_dir = base / case_name
            if case_dir.is_dir():
                for ext in extensions:
                    count += len(list(case_dir.glob(f"*{ext}")))
        return count
    else:
        return len(select_list)  # flat: select = stems, 1:1


def build_task(task_name: str, paths: dict, dry_run: bool = False) -> dict:
    """
    Procesa una tarea completa (breast o colorectal).
    Retorna {dataset_name: {"select": [...], "n_ids": N, "n_select": M, "n_slides": S}}
    """
    task_cfg = TASKS[task_name]
    meta_base = Path(paths["metadata_clean"])
    summary = {}

    for ds_cfg in task_cfg["datasets"]:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_cfg['name']}")
        print(f"  CSVs: {ds_cfg['csv_patterns']}")

        # 1. Leer y unificar IDs de todos los CSVs
        all_ids = set()
        for csv_rel in ds_cfg["csv_patterns"]:
            csv_path = meta_base / csv_rel
            if not csv_path.exists():
                print(f"  WARNING: CSV no encontrado: {csv_path}")
                continue
            ids = read_csv_ids(str(csv_path), ds_cfg["id_column"])
            print(f"  {csv_rel}: {len(ids)} IDs")
            all_ids.update(ids)

        print(f"  Total únicos (tras dedup): {len(all_ids)}")

        # 2. Resolver select list
        select_list = resolve_select_ids(ds_cfg, all_ids, paths)
        print(f"  Select final: {len(select_list)} elementos")

        if select_list and len(select_list) <= 5:
            print(f"  Valores: {select_list}")
        elif select_list:
            print(f"  Primeros 5: {select_list[:5]} ...")

        # 3. Contar slides reales si el directorio existe
        slides_dir = paths.get(ds_cfg["slides_path_key"], "")
        n_slides = count_slides_in_select(
            select_list, slides_dir, ds_cfg["subfolder_structure"], ds_cfg["format"])
        if n_slides >= 0:
            print(f"  Slides reales en disco: {n_slides}")
        else:
            print(f"  (No se pudo contar — directorio no accesible)")

        summary[ds_cfg["name"]] = {
            "select": select_list,
            "n_ids_csv": len(all_ids),
            "n_select": len(select_list),
            "n_slides": n_slides,
        }

        # 4. Generar YAMLs
        if not dry_run and select_list:
            patches_cfg = generate_patches_yaml(ds_cfg, select_list, task_name, paths)
            embed_cfg = generate_embeddings_yaml(ds_cfg, task_name, paths)

            # Guardar patches YAML
            patches_file = Path("config_files") / f"patches_{ds_cfg['name'].lower()}_filtered.yaml"
            patches_file.parent.mkdir(parents=True, exist_ok=True)
            with open(patches_file, "w") as f:
                yaml.dump(patches_cfg, f, default_flow_style=False, sort_keys=False,
                          allow_unicode=True)
            print(f"  → {patches_file}")

            # Guardar embeddings YAML
            embed_file = Path("config_files") / f"embeddings_{ds_cfg['name'].lower()}_filtered.yaml"
            with open(embed_file, "w") as f:
                yaml.dump(embed_cfg, f, default_flow_style=False, sort_keys=False,
                          allow_unicode=True)
            print(f"  → {embed_file}")

    return summary


def print_global_summary(all_summaries: dict):
    """Imprime resumen global de todas las tareas."""
    print(f"\n{'═'*70}")
    print("  RESUMEN GLOBAL")
    print(f"{'═'*70}")

    total_cases = 0
    total_slides = 0
    for task_name, datasets in all_summaries.items():
        print(f"\n  Tarea: {task_name.upper()}")
        for ds_name, info in datasets.items():
            n_slides = info.get("n_slides", -1)
            slides_str = f"{n_slides:>6d}" if n_slides >= 0 else "   N/A"
            print(f"    {ds_name:30s}  Cases: {info['n_select']:>5d}  Slides: {slides_str}")
            total_cases += info["n_select"]
            if n_slides >= 0:
                total_slides += n_slides

    print(f"\n  Total cases/directorios: {total_cases}")
    if total_slides > 0:
        print(f"  Total slides reales:     {total_slides}")
    print(f"{'═'*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera configs YAML filtrados desde metadata_clean/")
    parser.add_argument("--task", choices=["breast", "colorectal", "all"],
                        default="all", help="Tarea a procesar")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra qué se generaría, sin escribir archivos")
    args = parser.parse_args()

    print("=" * 70)
    print("  BUILD FILTERED CONFIGS")
    print(f"  Metadata: {PATHS['metadata_clean']}")
    if args.dry_run:
        print("  *** DRY RUN — no se escribirán archivos ***")
    print("=" * 70)

    tasks_to_run = ["breast", "colorectal"] if args.task == "all" else [args.task]
    all_summaries = {}

    for task in tasks_to_run:
        print(f"\n{'━'*70}")
        print(f"  TAREA: {task.upper()}")
        print(f"{'━'*70}")
        all_summaries[task] = build_task(task, PATHS, dry_run=args.dry_run)

    print_global_summary(all_summaries)


if __name__ == "__main__":
    main()
