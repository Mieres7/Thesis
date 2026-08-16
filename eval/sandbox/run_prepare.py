"""
eval/sandbox/run_prepare.py

Genera archivos .pkl (metadata + embeddings unidos) para cualquier
combinación de modelo × tarea.

Uso:
    cd eval/sandbox

    # Una combinación específica
    python run_prepare.py --task breast_molsub --model virchow2

    # Todas las tareas con un modelo
    python run_prepare.py --model virchow2 --all-tasks

    # Todo: todos los modelos × todas las tareas
    python run_prepare.py --all

    # Listar combinaciones disponibles
    python run_prepare.py --list
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.config.datasets import TASKS, MODELS
from eval.datasets.metadata_loader import load_all_metadata
from eval.datasets.embedding_loader import load_all_embeddings, join_metadata_and_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def prepare(task_key: str, model_name: str, output_dir: str = ".", tag: str = "") -> Path | None:
    """Carga metadata + embeddings, hace join, guarda .pkl.

    Si se especifica tag, guarda en un subdirectorio con ese nombre.
    Esto permite preparación parcial en servidores distintos sin
    conflictos de nombres.
    """
    task = TASKS[task_key]
    specs = task["datasets"]
    task_name = task["task_name"]

    pkl_name = f"final_df_{model_name}_{task_key}.pkl"
    if tag:
        pkl_dir = Path(output_dir) / tag
    else:
        pkl_dir = Path(output_dir)
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = pkl_dir / pkl_name

    logger.info("=" * 60)
    logger.info("PREPARANDO: task=%s | model=%s", task_name, model_name)
    logger.info("Datasets: %s", [s.name for s in specs])
    logger.info("Output: %s", pkl_path)
    logger.info("=" * 60)

    # 1. Metadata
    metadata_df = load_all_metadata(specs)
    logger.info("Metadata: %d pacientes con label", len(metadata_df))

    # 2. Embeddings
    embeddings_df = load_all_embeddings(specs, model_name=model_name)
    logger.info("Embeddings: %d pacientes", len(embeddings_df))

    # 3. Join
    final_df = join_metadata_and_embeddings(metadata_df, embeddings_df)
    logger.info("Final: %d pacientes", len(final_df))

    if len(final_df) == 0:
        logger.error("NO HAY PACIENTES tras el join — algo falló (¿embeddings faltantes?)")
        return None

    # 4. Guardar
    with open(pkl_path, "wb") as f:
        pickle.dump(final_df, f)
    logger.info("Guardado: %s (%d pacientes)", pkl_path, len(final_df))

    return pkl_path


def main():
    parser = argparse.ArgumentParser(description="Preparar .pkl de embeddings para evaluación")
    parser.add_argument("--task", choices=list(TASKS.keys()), help="Task key (ej: breast_molsub)")
    parser.add_argument("--model", choices=MODELS, help="Model name (ej: virchow2)")
    parser.add_argument("--all-tasks", action="store_true", help="Correr todas las tareas con el modelo dado")
    parser.add_argument("--all", action="store_true", help="Correr todas las combinaciones")
    parser.add_argument("--list", action="store_true", help="Listar combinaciones disponibles")
    parser.add_argument("--output-dir", default=".", help="Directorio de salida para .pkl")
    parser.add_argument("--tag", default="", help="Sufijo para el .pkl (ej: _munay, _prime). Permite preparación parcial por servidor.")
    args = parser.parse_args()

    if args.list:
        print("\nTareas disponibles:")
        for key, task in TASKS.items():
            datasets = [s.name for s in task["datasets"]]
            print(f"  {key:<20} task={task['task_name']:<25} datasets={datasets}")
        print(f"\nModelos: {MODELS}")
        print(f"\nTotal combinaciones: {len(TASKS)} tareas × {len(MODELS)} modelos = {len(TASKS) * len(MODELS)}")
        return

    if args.all:
        combos = [(t, m) for t in TASKS for m in MODELS]
    elif args.all_tasks and args.model:
        combos = [(t, args.model) for t in TASKS]
    elif args.task and args.model:
        combos = [(args.task, args.model)]
    else:
        parser.print_help()
        return

    print(f"\n{len(combos)} combinaciones a preparar:\n")
    for task_key, model in combos:
        print(f"  • {task_key} + {model}")
    print()

    results = []
    for task_key, model in combos:
        try:
            pkl_path = prepare(task_key, model, output_dir=args.output_dir, tag=args.tag)
            results.append((task_key, model, "OK" if pkl_path else "FAIL"))
        except Exception as e:
            logger.error("FALLÓ %s + %s: %s", task_key, model, e)
            results.append((task_key, model, f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for task_key, model, status in results:
        print(f"  {task_key:<20} {model:<15} {status}")


if __name__ == "__main__":
    main()
