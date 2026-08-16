"""
eval/sandbox/run_center_effect.py

Corre el experimento de center effect (clasificador de origen de dataset)
sobre los .pkl consolidados de embeddings a nivel de paciente que YA genera
el pipeline del Capítulo 6 — NO re-extrae parches ni embeddings.

Protocolo (idéntico al de la evaluación ID):
    MLPClassifier(256, ReLU, alpha=1e-4, max_iter=500) con sobremuestreo de
    clases minoritarias, StratifiedGroupKFold k=5 agrupado por paciente (uid),
    seed=42, bootstrap de 1000 remuestreos a nivel de paciente para IC 95%.
    Métricas: BA, macro-F1 y AUC (one-vs-rest para multiclase).

Experimento A — multiclase por dominio (etiqueta = dataset de origen):
    Mama (3 clases): BCNB vs HISTAI-Breast vs HSI-BC
    Colorrectal (3 clases): SurGen vs HISTAI-CRC-B1 vs HISTAI-CRC-B2

Experimento B — pares binarios en colorrectal (aislar el escáner):
    B1 vs B2 (mismo escáner) | SurGen vs B1 | SurGen vs B2

Uso:
    cd eval/sandbox
    python run_center_effect.py                          # A y B, 4 modelos
    python run_center_effect.py --exp A                  # solo multiclase
    python run_center_effect.py --exp B                  # solo pares binarios
    python run_center_effect.py --models uni2,virchow2
    python run_center_effect.py --breast-task breast_her2 --crc-task crc_side

Salidas (en --output-dir):
    center_effect_{key}_{model}.json  → resultado completo por combinación
    center_effect_summary.csv         → tabla consolidada (para Apéndice F)
    center_effect_summary.md          → tabla markdown imprimible
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.config.datasets import MODELS
from eval.evaluation.center_effect import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    ci_string,
    comparison_label,
    center_effect_classify,
    summary_row,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 1000


def build_experiments(exp_filter: str, breast_task: str, crc_task: str) -> list[dict]:
    """Lista de experimentos con el task_key de origen ya resuelto."""
    experiments = []
    if exp_filter in ("A", "both"):
        experiments.extend(EXPERIMENT_A)
    if exp_filter in ("B", "both"):
        experiments.extend(EXPERIMENT_B)
    resolved = []
    for exp in experiments:
        if exp["domain"] == "breast":
            resolved.append({**exp, "task_key": breast_task})
        elif exp["domain"] == "colorectal":
            resolved.append({**exp, "task_key": crc_task})
        else:
            resolved.append(dict(exp))
    return resolved


def load_pkl_cached(pkl_path: Path, cache: dict) -> pd.DataFrame:
    """Carga (y cachea) un .pkl consolidado: el mismo pkl sirve a varios
    experimentos (p. ej. A_colorectal + 3 pares binarios usan crc_site)."""
    key = str(pkl_path)
    if key not in cache:
        with open(pkl_path, "rb") as f:
            cache[key] = pickle.load(f)
        logger.info("Cargado pkl: %s (%d pacientes)", pkl_path.name, len(cache[key]))
    return cache[key]


def markdown_table(rows: list[dict]) -> str:
    header = ["dominio", "comparación", "modelo", "n_test", "BA [IC95]", "macro-F1 [IC95]", "AUC"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        auc = r["auc"]
        auc_str = "—" if auc is None else f"{auc:.3f}"
        lines.append(
            "| " + " | ".join([
                r["domain_display"],
                r["comparison"],
                r["model"],
                str(r["n_test"]),
                r["ba_ci"],
                r["f1_ci"],
                auc_str,
            ]) + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Center effect: clasificador de origen de dataset")
    parser.add_argument("--pkl-dir", default="pkl_final", help="Directorio con los .pkl consolidados")
    parser.add_argument("--output-dir", default="results_center_effect", help="Directorio de salida")
    parser.add_argument("--models", default=",".join(MODELS), help="Modelos separados por coma (default: todos)")
    parser.add_argument("--exp", choices=["A", "B", "both"], default="both",
                        help="Experimento(s) a correr (default: both)")
    parser.add_argument("--breast-task", choices=["breast_molsub", "breast_her2"],
                        default="breast_molsub", help="Task del pkl para el dominio mama")
    parser.add_argument("--crc-task", choices=["crc_site", "crc_side"],
                        default="crc_site", help="Task del pkl para el dominio colorrectal")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed global (default: 42)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="Remuestreos bootstrap (default: 1000)")
    args = parser.parse_args()

    models = [m for m in args.models.split(",") if m]
    pkl_dir = Path(args.pkl_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(args.exp, args.breast_task, args.crc_task)

    print("\n" + "=" * 72)
    print("CENTER EFFECT — clasificador de origen de dataset")
    print(f"Protocolo: MLP(256, alpha=1e-4, max_iter=500) + oversampling | "
          f"StratifiedGroupKFold k=5 por uid, seed={args.seed} | "
          f"bootstrap {args.n_bootstrap} (IC 95%)")
    print("=" * 72)

    rows = []
    pkl_cache = {}
    missing = []

    for model in models:
        for exp in experiments:
            pkl_path = pkl_dir / f"final_df_{model}_{exp['task_key']}.pkl"
            if not pkl_path.exists():
                missing.append((exp["key"], model, exp["task_key"]))
                continue

            logger.info("=" * 60)
            logger.info("EXPERIMENTO %s | %s | modelo=%s | comparación: %s",
                        exp["experiment"], exp["key"], model, comparison_label(exp["datasets"]))
            logger.info("=" * 60)

            df = load_pkl_cached(pkl_path, pkl_cache)
            result = center_effect_classify(
                df,
                exp["datasets"],
                seed=args.seed,
                n_bootstrap=args.n_bootstrap,
            )

            payload = {
                "experiment": exp["experiment"],
                "key": exp["key"],
                "domain": exp["domain"],
                "comparison": comparison_label(exp["datasets"]),
                "datasets": exp["datasets"],
                "task_key": exp["task_key"],
                "note": exp.get("note"),
                "model": model,
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "protocol": {
                    "probe": "MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4, max_iter=500, early_stopping=False)",
                    "oversampling": "minority classes to max class size (fit_mlp_probe balanced=True)",
                    "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True) grouped by uid",
                    "bootstrap_level": "patient (uid)",
                    "auc": "one-vs-rest macro (multiclass) / ROC AUC (binary)",
                },
                "n_total": result["n_total"],
                "n_test": result["n_test"],
                "classes": result["classes"],
                "test_metrics": result["test_metrics"],
                "bootstrap": result["bootstrap"],
                "confusion_matrix": result["confusion_matrix"],
                "per_fold": result["per_fold"],
            }
            json_path = out_dir / f"center_effect_{exp['key']}_{model}.json"
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info("Guardado: %s", json_path)

            rows.append(summary_row(exp, model, result))

            t = result["test_metrics"]
            print(f"  [{exp['experiment']}] {comparison_label(exp['datasets']):<45} "
                  f"n_test={result['n_test']:<5} "
                  f"BA={ci_string(result['bootstrap'], 'balanced_accuracy')}  "
                  f"F1={ci_string(result['bootstrap'], 'f1_macro')}  "
                  f"AUC={t.get('auc', t.get('auc_ovr_macro')):.3f}")
        print()

    if missing:
        print("Faltan .pkl (se omiten):")
        for key, model, task in missing:
            print(f"  • {key} + {model} -> final_df_{model}_{task}.pkl")

    if not rows:
        print("No se ejecutó ningún experimento (¿pkl-dir incorrecto?).")
        sys.exit(1)

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "center_effect_summary.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Resumen CSV: %s", csv_path)

    md = markdown_table(rows)
    md_path = out_dir / "center_effect_summary.md"
    md_path.write_text(md + "\n", encoding="utf-8")
    logger.info("Resumen markdown: %s", md_path)

    print("\n" + "=" * 72)
    print("TABLA CONSOLIDADA (dominio | comparación | modelo | n_test | BA [IC95] | macro-F1 [IC95] | AUC)")
    print("=" * 72)
    print(md)


if __name__ == "__main__":
    main()