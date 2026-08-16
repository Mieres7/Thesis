"""
eval/sandbox/run_merge.py

Mergea archivos .pkl generados con --tag en distintos servidores,
produciendo el .pkl final unificado listo para run_eval.py.

Los .pkl parciales están en subdirectorios por tag:
    ./munay/final_df_uni2_breast_her2.pkl
    ./prime/final_df_uni2_breast_her2.pkl  →  final_df_uni2_breast_her2.pkl

Uso:
    python run_merge.py --input-dir . --output-dir .

    # Solo ciertos tags
    python run_merge.py --tags munay,prime

    # Modo dry-run
    python run_merge.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import pickle
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def find_tag_dirs(input_dir: str, tags: list[str] | None) -> list[Path]:
    """Busca subdirectorios que contengan .pkl (cada uno = un tag)."""
    base = Path(input_dir)
    tag_dirs = []
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        if tags and item.name not in tags:
            continue
        # Solo directorios que contengan .pkl
        if list(item.glob("final_df_*.pkl")):
            tag_dirs.append(item)
    return tag_dirs


def find_pkls_in_tags(tag_dirs: list[Path]) -> dict[tuple[str, str], list[Path]]:
    """
    Escanea los directorios tag.
    Retorna dict: (model, task) -> [lista de paths ordenada por tag]
    """
    partials: dict[tuple[str, str], list[Path]] = {}

    for tag_dir in tag_dirs:
        for pkl_path in sorted(tag_dir.glob("final_df_*.pkl")):
            name = pkl_path.name  # final_df_{model}_{task}.pkl
            stem = name.replace("final_df_", "").replace(".pkl", "")  # {model}_{task}
            # El último _ separa model de task
            # Pero model puede tener _ (phikon_v2). Usamos los modelos conocidos.
            # Estrategia: split por _ y recomponer
            parts = stem.split("_")
            # Los nombres de modelo conocidos: uni2, virchow2, phikon_v2, prov_gigapath
            # Los nombres de tarea conocidos: breast_her2, breast_molsub, crc_site, crc_side, organ
            known_models = ["uni2", "virchow2", "phikon_v2", "prov_gigapath"]
            known_tasks = ["breast_her2", "breast_molsub", "crc_site", "crc_side", "organ"]

            model = None
            task = None

            # Probar cada modelo conocido como prefijo
            for m in known_models:
                if stem.startswith(m + "_"):
                    suffix = stem[len(m) + 1:]  # remueve "model_"
                    # Verificar que el sufijo sea una tarea conocida
                    if suffix in known_tasks:
                        model = m
                        task = suffix
                        break

            if model is None or task is None:
                logger.warning("No se pudo parsear: %s", name)
                continue

            key = (model, task)
            partials.setdefault(key, []).append(pkl_path)

    return partials


def merge_and_save(
    key: tuple[str, str],
    partial_paths: list[Path],
    output_dir: str,
    dry_run: bool = False,
) -> bool:
    """Mergea los .pkl parciales para un (model, task) y guarda el unificado."""
    model, task = key
    output_path = Path(output_dir) / f"final_df_{model}_{task}.pkl"

    if dry_run:
        sources = ", ".join(f"{p.parent.name}/{p.name}" for p in partial_paths)
        logger.info("[DRY-RUN] %s -> %s  (from: %s)", key, output_path.name, sources)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for pkl_path in partial_paths:
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        datasets = df["dataset"].unique().tolist() if "dataset" in df.columns else ["?"]
        logger.info("  + %s/%s: %d pacientes (%s)", pkl_path.parent.name, pkl_path.name, len(df), datasets)
        frames.append(df)

    if not frames:
        logger.warning("No data for %s", key)
        return False

    merged = pd.concat(frames, ignore_index=True)

    n_dupes = merged["uid"].duplicated().sum()
    if n_dupes > 0:
        logger.warning(
            "%d uids duplicados en merge %s — mismos pacientes en ambos servidores?",
            n_dupes, key,
        )
        merged = merged.drop_duplicates(subset=["uid"])

    with open(output_path, "wb") as f:
        pickle.dump(merged, f)

    logger.info(
        "GUARDADO: %s (%d pacientes, %d fuentes)",
        output_path.name, len(merged), len(frames),
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Mergear .pkl parciales de run_prepare.py")
    parser.add_argument("--input-dir", default=".", help="Directorio con subdirectorios tag")
    parser.add_argument("--output-dir", default="pkl_final", help="Directorio de salida para .pkl unificados")
    parser.add_argument("--tags", default=None, help="Tags a mergear (separados por coma). Default: todos.")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar qué se va a mergear")
    args = parser.parse_args()

    tags = args.tags.split(",") if args.tags else None
    tag_dirs = find_tag_dirs(args.input_dir, tags)

    if not tag_dirs:
        logger.error("No se encontraron directorios tag con .pkl en %s", args.input_dir)
        return

    logger.info("Tags encontrados: %s", [d.name for d in tag_dirs])

    partials = find_pkls_in_tags(tag_dirs)

    if not partials:
        logger.error("No se encontraron .pkl parciales válidos")
        return

    print(f"\n{'='*60}")
    print(f"MERGEANDO {len(partials)} combinaciones desde {len(tag_dirs)} tags")
    print(f"{'='*60}\n")

    ok = 0
    fail = 0
    for key, paths in sorted(partials.items()):
        if len(paths) == 1 and not args.dry_run:
            model, task = key
            single_path = paths[0]
            output_path = Path(args.output_dir) / f"final_df_{model}_{task}.pkl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if single_path != output_path:
                shutil.copy2(single_path, output_path)
                logger.info("COPIADO (un solo tag): %s -> %s", single_path.name, output_path.name)
            else:
                logger.info("SALTADO (ya es destino): %s", single_path.name)
            ok += 1
            continue

        if merge_and_save(key, paths, args.output_dir, dry_run=args.dry_run):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"RESUMEN: {ok} OK, {fail} FAIL")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
