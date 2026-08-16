"""
eval/sandbox/run_all.py

Orquestador del pipeline completo de evaluación.
Detecta automáticamente el servidor donde se ejecuta y corre
los pasos correspondientes.

Uso:
    # Pipeline completo (prepare → merge → eval)
    python run_all.py

    # Solo un paso
    python run_all.py --step prepare
    python run_all.py --step merge
    python run_all.py --step eval

    # Especificar tag manualmente (default: detecta hostname)
    python run_all.py --tag munay

    # Skip cross-dataset evaluation
    python run_all.py --skip-cross-dataset

    # Solo ciertos modelos/tareas
    python run_all.py --models uni2,virchow2
    python run_all.py --tasks breast_her2,crc_site
"""
from __future__ import annotations

import argparse
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")

# Hostname → tag mapping para preparación parcial
# Ajustar según los nombres reales de los servidores
HOSTNAME_TAGS = {
    "munay": "munay",
    "prime": "prime",
    # Agregar más servidores si es necesario
}

DEFAULT_TAG = "local"


def detect_tag() -> str:
    """Detecta el servidor actual por hostname."""
    import socket
    hostname = socket.gethostname().lower().split(".")[0]
    for key, tag in HOSTNAME_TAGS.items():
        if key in hostname:
            return tag
    logger.warning("Hostname '%s' no reconocido, usando tag='%s'", hostname, DEFAULT_TAG)
    return DEFAULT_TAG


def step_prepare(tag: str, models: list[str] | None, tasks: list[str] | None):
    """Corre run_prepare.py con el tag del servidor actual."""
    logger.info("=" * 60)
    logger.info("PASO 1: PREPARE (tag=%s)", tag)
    logger.info("=" * 60)

    cmd = [sys.executable, "run_prepare.py", "--all", "--tag", tag]
    if models:
        cmd = [sys.executable, "run_prepare.py", "--tag", tag] + \
              [a for m in models for a in ("--model", m, "--all-tasks")]
    if tasks:
        cmd += ["--tasks"] + tasks

    logger.info("Ejecutando: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    logger.info("Prepare completado en %.1f min (exit code=%d)", elapsed / 60, result.returncode)
    return result.returncode == 0


def step_merge(tag: str):
    """Mergea los .pkl parciales del servidor actual con los existentes."""
    logger.info("=" * 60)
    logger.info("PASO 2: MERGE")
    logger.info("=" * 60)

    merge_script = Path(__file__).parent / "run_merge.py"
    cmd = [sys.executable, str(merge_script), "--input-dir", ".", "--output-dir", "."]

    logger.info("Ejecutando: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    logger.info("Merge completado en %.1f seg (exit code=%d)", elapsed, result.returncode)
    return result.returncode == 0


def step_eval(models: list[str] | None, tasks: list[str] | None, cross_dataset: bool = True, output_dir: str = "results"):
    """Corre la evaluación completa."""
    logger.info("=" * 60)
    logger.info("PASO 3: EVAL" + (" (con cross-dataset)" if cross_dataset else ""))
    logger.info("Output dir: %s", output_dir)
    logger.info("=" * 60)

    cmd = [sys.executable, "run_eval.py", "--all", "--output-dir", output_dir, "--no-npz"]
    if cross_dataset:
        cmd.append("--cross-dataset")
    if models:
        pass  # --all ya corre todos los modelos
    if tasks:
        pass  # --all ya corre todas las tareas

    logger.info("Ejecutando: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    logger.info("Eval completado en %.1f min (exit code=%d)", elapsed / 60, result.returncode)
    return result.returncode == 0


def check_ready() -> bool:
    """Verifica que todos los .pkl finales existan en pkl_final/."""
    from eval.config.datasets import TASKS, MODELS

    pkl_dir = Path(__file__).parent / "pkl_final"
    missing = []
    for task_key in TASKS:
        for model in MODELS:
            pkl_path = pkl_dir / f"final_df_{model}_{task_key}.pkl"
            if not pkl_path.exists():
                missing.append((task_key, model))

    if missing:
        logger.warning("Faltan %d .pkl finales (correr --step merge primero):", len(missing))
        for t, m in missing:
            logger.warning("  • %s + %s", t, m)
        return False
    logger.info("Todos los .pkl finales están listos (%d combinaciones)", len(TASKS) * len(MODELS))
    return True


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de evaluación de tesis")
    parser.add_argument("--step", choices=["prepare", "merge", "eval", "all"], default="all",
                        help="Paso(s) a ejecutar (default: all)")
    parser.add_argument("--tag", default=None,
                        help="Tag del servidor actual (default: detecta por hostname)")
    parser.add_argument("--skip-cross-dataset", action="store_true",
                        help="Saltar evaluación cross-dataset")
    parser.add_argument("--output-dir", default="results",
                        help="Directorio de salida para los JSON/CSV/npz de evaluación")
    parser.add_argument("--models", default=None,
                        help="Modelos a evaluar (separados por coma). Default: todos.")
    parser.add_argument("--tasks", default=None,
                        help="Tareas a evaluar (separadas por coma). Default: todas.")
    args = parser.parse_args()

    tag = args.tag or detect_tag()
    models = args.models.split(",") if args.models else None
    tasks = args.tasks.split(",") if args.tasks else None

    all_ok = True

    if args.step in ("prepare", "all"):
        ok = step_prepare(tag, models, tasks)
        all_ok = all_ok and ok

    if args.step in ("merge", "all"):
        ok = step_merge(tag)
        all_ok = all_ok and ok

    if args.step in ("eval", "all"):
        ready = check_ready()
        if not ready:
            logger.error("Faltan .pkl finales. Corré --step merge primero.")
            all_ok = False
        else:
            ok = step_eval(models, tasks, cross_dataset=not args.skip_cross_dataset,
                           output_dir=args.output_dir)
            all_ok = all_ok and ok

    if all_ok:
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("PIPELINE COMPLETADO CON ERRORES — revisar logs")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
