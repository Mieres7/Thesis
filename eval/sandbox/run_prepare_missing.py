"""
eval/sandbox/run_prepare_missing.py

Genera archivos .pkl (metadata + embeddings unidos) para combinaciones
modelo × tarea.

Uso:
cd eval/sandbox

# Reejecución completa: corre todas las combinaciones excepto las marcadas
# como ya completadas en COMPLETED_COMBOS.
python run_prepare_missing.py --all --skip-completed --output-dir . --tag prime

# Ver qué combinaciones correría el comando anterior, sin ejecutarlas.
python run_prepare_missing.py --all --skip-completed --dry-run --output-dir . --tag prime

# Una combinación específica.
python run_prepare_missing.py --task crc_side --model phikon_v2 --output-dir . --tag prime

# Varias combinaciones específicas.
python run_prepare_missing.py \
  --combo crc_side:phikon_v2 \
  --combo crc_site:uni2 \
  --output-dir . --tag prime

# Saltar cualquier PKL que ya exista físicamente en el directorio de salida.
python run_prepare_missing.py --all --skip-existing --output-dir . --tag prime
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.config.datasets import MODELS, TASKS
from eval.datasets.embedding_loader import load_all_embeddings, join_metadata_and_embeddings
from eval.datasets.metadata_loader import load_all_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Combinaciones que aparecen marcadas con "M" como listas en la captura.
# Formato: (task_key, model_name).
COMPLETED_COMBOS: set[tuple[str, str]] = {
    ("breast_her2", "phikon_v2"),
    ("breast_molsub", "phikon_v2"),
    ("breast_her2", "prov_gigapath"),
    ("breast_molsub", "prov_gigapath"),
    ("breast_her2", "uni2"),
    ("breast_molsub", "uni2"),
    ("organ", "uni2"),
    ("breast_molsub", "virchow2"),
}


def output_path(task_key: str, model_name: str, output_dir: str, tag: str) -> Path:
    pkl_dir = Path(output_dir) / tag if tag else Path(output_dir)
    return pkl_dir / f"final_df_{model_name}_{task_key}.pkl"


def prepare(task_key: str, model_name: str, output_dir: str = ".", tag: str = "") -> Path | None:
    """Carga metadata + embeddings, hace join y guarda el PKL."""
    task = TASKS[task_key]
    specs = task["datasets"]
    task_name = task["task_name"]
    pkl_path = output_path(task_key, model_name, output_dir, tag)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PREPARANDO: task=%s | model=%s", task_name, model_name)
    logger.info("Datasets: %s", [spec.name for spec in specs])
    logger.info("Output: %s", pkl_path)
    logger.info("=" * 60)

    start = time.perf_counter()
    metadata_df = load_all_metadata(specs)
    logger.info(
        "Metadata: %d pacientes con label | %.1f s",
        len(metadata_df),
        time.perf_counter() - start,
    )

    start = time.perf_counter()
    embeddings_df = load_all_embeddings(specs, model_name=model_name)
    logger.info(
        "Embeddings: %d pacientes | %.1f s",
        len(embeddings_df),
        time.perf_counter() - start,
    )

    start = time.perf_counter()
    final_df = join_metadata_and_embeddings(metadata_df, embeddings_df)
    logger.info("Final: %d pacientes | join: %.1f s", len(final_df), time.perf_counter() - start)

    if len(final_df) == 0:
        logger.error("NO HAY PACIENTES tras el join — ¿faltan embeddings o metadata?")
        return None

    start = time.perf_counter()
    with open(pkl_path, "wb") as file:
        pickle.dump(final_df, file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Guardado: %s | %.1f s", pkl_path, time.perf_counter() - start)

    return pkl_path


def parse_combos(values: list[str], parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for value in values:
        try:
            task_key, model_name = value.split(":", 1)
        except ValueError:
            parser.error(f"Formato inválido: {value!r}. Usa TASK:MODEL, por ejemplo crc_side:phikon_v2")

        if task_key not in TASKS:
            parser.error(f"Task desconocida: {task_key}. Ejecuta --list para ver las disponibles.")
        if model_name not in MODELS:
            parser.error(f"Modelo desconocido: {model_name}. Ejecuta --list para ver los disponibles.")

        combos.append((task_key, model_name))
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparar PKL de embeddings para evaluación")
    parser.add_argument("--task", choices=list(TASKS.keys()), help="Task key, ej. breast_molsub")
    parser.add_argument("--model", choices=MODELS, help="Modelo, ej. virchow2")
    parser.add_argument("--combo", action="append", default=[], metavar="TASK:MODEL", help="Combinación específica; repetible")
    parser.add_argument("--all-tasks", action="store_true", help="Correr todas las tareas con el modelo indicado")
    parser.add_argument("--all", action="store_true", help="Correr todas las combinaciones task × modelo")
    parser.add_argument("--list", action="store_true", help="Listar tareas, modelos y combinaciones marcadas como completadas")
    parser.add_argument("--output-dir", default=".", help="Directorio raíz para los PKL")
    parser.add_argument("--tag", default="", help="Subdirectorio de salida, ej. prime")
    parser.add_argument("--skip-completed", action="store_true", help="Saltar solo las combinaciones definidas en COMPLETED_COMBOS")
    parser.add_argument("--skip-existing", action="store_true", help="Saltar combinaciones cuyo archivo PKL ya exista en la salida")
    parser.add_argument("--overwrite", action="store_true", help="No aplicar --skip-existing; permite reemplazar PKL existentes")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar el plan sin cargar datos ni escribir archivos")
    args = parser.parse_args()

    if args.list:
        print("\nTareas disponibles:")
        for key, task in TASKS.items():
            datasets = [spec.name for spec in task["datasets"]]
            print(f" {key:<20} task={task['task_name']:<25} datasets={datasets}")
        print(f"\nModelos: {MODELS}")
        print("\nCombinaciones configuradas como completadas:")
        for task_key, model_name in sorted(COMPLETED_COMBOS):
            print(f" {task_key:<20} {model_name}")
        return

    selected_modes = sum(bool(value) for value in (args.all, args.all_tasks, args.combo, args.task or args.model))
    if selected_modes != 1:
        parser.error("Usa exactamente una selección: --all, --all-tasks --model, --combo o --task --model.")

    if args.all:
        combos = [(task_key, model_name) for task_key in TASKS for model_name in MODELS]
    elif args.all_tasks:
        if not args.model:
            parser.error("--all-tasks requiere --model.")
        combos = [(task_key, args.model) for task_key in TASKS]
    elif args.combo:
        combos = parse_combos(args.combo, parser)
    else:
        if not (args.task and args.model):
            parser.error("--task requiere --model, y viceversa.")
        combos = [(args.task, args.model)]

    planned: list[tuple[str, str]] = []
    skipped: list[tuple[str, str, str]] = []
    for task_key, model_name in combos:
        pkl_path = output_path(task_key, model_name, args.output_dir, args.tag)
        if args.skip_completed and (task_key, model_name) in COMPLETED_COMBOS:
            skipped.append((task_key, model_name, "marcado como completado"))
        elif args.skip_existing and not args.overwrite and pkl_path.exists():
            skipped.append((task_key, model_name, f"ya existe: {pkl_path}"))
        else:
            planned.append((task_key, model_name))

    print(f"\nPlan: {len(planned)} combinaciones por ejecutar; {len(skipped)} omitidas.\n")
    if skipped:
        print("OMITIDAS:")
        for task_key, model_name, reason in skipped:
            print(f" - {task_key:<20} {model_name:<18} {reason}")
        print()

    if planned:
        print("A EJECUTAR:")
        for task_key, model_name in planned:
            print(f" - {task_key:<20} {model_name}")
        print()
    else:
        print("No quedan combinaciones por ejecutar.")
        return

    if args.dry_run:
        print("Dry run: no se ejecutó ninguna combinación.")
        return

    results: list[tuple[str, str, str]] = []
    for task_key, model_name in planned:
        try:
            pkl_path = prepare(task_key, model_name, output_dir=args.output_dir, tag=args.tag)
            results.append((task_key, model_name, "OK" if pkl_path else "FAIL"))
        except Exception as error:
            logger.exception("FALLÓ %s + %s", task_key, model_name)
            results.append((task_key, model_name, f"ERROR: {error}"))

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for task_key, model_name, status in results:
        print(f" {task_key:<20} {model_name:<18} {status}")


if __name__ == "__main__":
    main()
