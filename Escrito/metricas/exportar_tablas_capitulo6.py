#!/usr/bin/env python3
"""
Escrito/metricas/exportar_tablas_capitulo6.py

Exporta a CSV los valores numéricos de las tablas de resultados del
Capítulo 6 (06-oe3.tex), calculando todo dinámicamente desde los JSON
generados por eval/sandbox/run_eval.py en el repositorio tesis.

FUENTES (todas en <tesis>/eval/sandbox/results/):
  - eval_<model>_<task>_<dataset>.json      → utilidad global in-distribution (ID)
                                             y fairness por subgrupos (sexo/edad).
  - cross_<model>_<task>_<train>_to_<test>_mlp.json → generalización out-of-distribution (OOD).

El "ID" de cada par cross-dataset se toma del eval del dataset de
entrenamiento (test_metrics del mismo dataset), tal como define el
Capítulo 6 en las ecuaciones Δ_OOD y Δ_rel.

Salidas (en el mismo directorio de este script):
  - tabla6_utilidad_global.csv            granular: tarea × modelo × dataset
  - tabla6_utilidad_global_agregada.csv   tarea × modelo (media no ponderada entre datasets)
  - tabla6_fairness_sex.csv               brechas por sexo (datasets con sexo)
  - tabla6_fairness_age.csv               brechas por grupo de edad
  - tabla6_cross_dataset.csv              pares train→test con caída ID vs OOD
  - fig_ba_por_tarea_modelo.png           (opcional, --figures)
  - fig_cross_ba_por_par.png              (opcional, --figures)

Uso:
    python exportar_tablas_capitulo6.py [--results-dir RUTA] [--output-dir RUTA] [--figures]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

MODELS = ["uni2", "virchow2", "phikon_v2", "prov_gigapath"]

MODEL_LABEL = {
    "uni2": "UNI2",
    "virchow2": "Virchow2",
    "phikon_v2": "Phikon-v2",
    "prov_gigapath": "Prov-GigaPath",
}

# Los JSON guardan el task_name (campo "task"), no la clave de TASKS.
TASK_LABEL = {
    "her2_status": "HER2",
    "molecular_subtype": "MolSub",
    "site": "Site",
    "side": "Side",
    "organ": "Organ",
}

# Clave del AUC según el tipo de tarea (binaria usa 'auc', multiclase 'auc_ovr_macro')
AUC_KEYS = ("auc", "auc_ovr_macro")


def _find_results_dir() -> Path:
    env = os.environ.get("TESIS_RESULTS_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    here = Path(__file__).resolve().parent  # .../Escrito/metricas
    candidates = [
        here.parent.parent.parent / "tesis" / "eval" / "sandbox" / "results",
        here.parent.parent / "tesis" / "eval" / "sandbox" / "results",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit(
        "No se encontró el directorio de resultados. Usá --results-dir "
        "o la variable de entorno TESIS_RESULTS_DIR."
    )


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt3(x) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return ""


def _fmt_pct(x) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.1f}"
    except (TypeError, ValueError):
        return ""


def _auc(metrics: dict):
    for k in AUC_KEYS:
        v = metrics.get(k)
        if v is not None:
            return float(v)
    return None


def _auc_key(metrics: dict) -> str:
    for k in AUC_KEYS:
        if metrics.get(k) is not None:
            return k
    return AUC_KEYS[0]


def _bootstrap_ci(bs: dict, key: str):
    entry = (bs or {}).get(key) or {}
    lo, hi = entry.get("ci_low"), entry.get("ci_high")
    if lo is None or hi is None:
        return None, None
    return float(lo), float(hi)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path} ({len(rows)} filas)")


# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES DE FAIRNESS
# ─────────────────────────────────────────────────────────────────────────────

def _fairness_rows(d: dict, dim: str) -> list[dict]:
    """Devuelve las filas por subgrupo embebidas en el JSON (fairness_<dim>)."""
    return d["probes"]["mlp"].get(f"fairness_{dim}") or []


def _gap_from_records(records: list[dict], metric: str):
    vals = [float(r[metric]) for r in records if r.get(metric) is not None]
    if len(vals) < 2:
        return None
    return max(vals) - min(vals)


def _fairness_summary_row(d: dict, dim: str, records: list[dict]) -> dict:
    fs = d["probes"]["mlp"].get("fairness_summary", {}).get(dim) or {}
    da = fs.get("delta_auc") or {}
    eo = fs.get("equalized_odds") or {}
    return {
        "task": TASK_LABEL.get(d["task"], d["task"]),
        "model": MODEL_LABEL.get(d["model"], d["model"]),
        "dataset": d["dataset"],
        "n_test": d["n_test"],
        "n_subgroups": fs.get("n_subgroups"),
        "gap_BA": _fmt3(_gap_from_records(records, "balanced_accuracy")),
        "gap_F1": _fmt3(_gap_from_records(records, "f1_macro")),
        "delta_AUC": _fmt3(da.get("delta_auc")),
        "best_auc_group": da.get("best_group") or "",
        "worst_auc_group": da.get("worst_group") or "",
        "AUC_ES": _fmt3(fs.get("auc_es")),
        "avg_gap": _fmt3((fs.get("avg_gap") or {}).get("avg_gap")),
        "tpr_gap": _fmt3(eo.get("tpr_gap")),
        "fpr_gap": _fmt3(eo.get("fpr_gap")),
        "tpr_best_group": eo.get("tpr_best_group") or "",
        "tpr_worst_group": eo.get("tpr_worst_group") or "",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TABLA 4 – Cross-dataset (ID vs OOD)
# ─────────────────────────────────────────────────────────────────────────────

def tabla_cross_dataset(results_dir: Path, id_index: dict) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("cross_*_mlp.json")):
        d = _load_json(path)
        tr, te = d["train_dataset"], d["test_dataset"]
        task = d["task"]
        key = (d["model"], task, tr)
        id_metrics = id_index.get(key)
        if id_metrics is None:
            print(f"  [aviso] sin ID para {d['model']}/{task}/{tr}", file=sys.stderr)
            continue
        ood = d["metrics"]
        ba_id = id_metrics["balanced_accuracy"]
        ba_ood = ood.get("balanced_accuracy")
        f1_id = id_metrics["f1_macro"]
        f1_ood = ood.get("f1_macro")
        rows.append({
            "task": TASK_LABEL.get(task, task),
            "model": MODEL_LABEL.get(d["model"], d["model"]),
            "train": tr,
            "test": te,
            "n_train": d["n_train"],
            "n_test": d["n_test"],
            "BA_ID": _fmt3(ba_id),
            "BA_OOD": _fmt3(ba_ood),
            "caida_BA": _fmt3(ba_id - ba_ood if ba_id is not None and ba_ood is not None else None),
            "caida_BA_rel_pct": _fmt_pct(
                (ba_id - ba_ood) / ba_id * 100 if ba_id not in (None, 0) and ba_ood is not None else None
            ),
            "F1_ID": _fmt3(f1_id),
            "F1_OOD": _fmt3(f1_ood),
            "caida_F1": _fmt3(f1_id - f1_ood if f1_id is not None and f1_ood is not None else None),
            "caida_F1_rel_pct": _fmt_pct(
                (f1_id - f1_ood) / f1_id * 100 if f1_id not in (None, 0) and f1_ood is not None else None
            ),
            "AUC_OOD": _fmt3(_auc(ood)),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURAS (opcional)
# ─────────────────────────────────────────────────────────────────────────────

def figuras(utilidad_agregada: list[dict], cross_rows: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figura 1: BA agregada por tarea × modelo
    tasks = sorted({r["task"] for r in utilidad_agregada})
    labels = [MODEL_LABEL[m] for m in MODELS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(tasks))
    width = 0.2
    for i, m in enumerate(MODELS):
        vals = []
        for t in tasks:
            r = next((r for r in utilidad_agregada if r["task"] == t and r["model_key"] == m), None)
            vals.append(r["BA_agregada"] if r else 0)
        ax.bar(x + (i - 1.5) * width, vals, width, label=labels[i])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Balanced accuracy (media entre datasets)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ba_por_tarea_modelo.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out_dir / 'fig_ba_por_tarea_modelo.png'}")

    # Figura 2: BA OOD por par train→test, una sub-figura por tarea
    tasks_cross = sorted({r["task"] for r in cross_rows})
    fig, axes = plt.subplots(1, len(tasks_cross), figsize=(4.5 * len(tasks_cross), 4.5), squeeze=False)
    for ax, t in zip(axes[0], tasks_cross):
        rows = [r for r in cross_rows if r["task"] == t]
        pairs = sorted({(r["train"], r["test"]) for r in rows})
        labels = [f"{p[0]}→{p[1]}" for p in pairs]
        x = np.arange(len(pairs))
        for i, m in enumerate(MODELS):
            vals = []
            for p in pairs:
                r = next(
                    (r for r in rows if r["model_key"] == m and (r["train"], r["test"]) == p),
                    None,
                )
                vals.append(float(r["BA_OOD"]) if r and r["BA_OOD"] else 0)
            ax.bar(x + (i - 1.5) * width, vals, width, label=MODEL_LABEL[m])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(t)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    handles, lbls = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, lbls, ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_dir / "fig_cross_ba_por_par.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out_dir / 'fig_cross_ba_por_par.png'}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exporta tablas del Capítulo 6 desde los resultados de eval/")
    parser.add_argument("--results-dir", default=None, help="Directorio con los JSON de resultados")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida (default: este script)")
    parser.add_argument("--figures", action="store_true", help="Generar figuras PNG")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _find_results_dir()
    if not results_dir.is_dir():
        raise SystemExit(f"No existe el directorio de resultados: {results_dir}")
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exportando tablas del Capítulo 6 ===")
    print(f"  Resultados: {results_dir}")
    print(f"  Salida:     {out_dir}\n")

    # ── Índice de utilidad ID para el cross-dataset ─────────────────────────
    id_index = {}
    granular = []
    for path in sorted(results_dir.glob("eval_*.json")):
        d = _load_json(path)
        tm = d["probes"]["mlp"]["test_metrics"]
        id_index[(d["model"], d["task"], d["dataset"])] = tm

    # ── Tabla 1: utilidad global granular ────────────────────────────────────
    print("[1/4] Utilidad global in-distribution (granular)")
    granular = []
    for path in sorted(results_dir.glob("eval_*.json")):
        d = _load_json(path)
        tm = d["probes"]["mlp"]["test_metrics"]
        bs = d["probes"]["mlp"]["bootstrap"]
        auc_v = _auc(tm)
        auc_lo, auc_hi = _bootstrap_ci(bs, _auc_key(tm))
        ba_lo, ba_hi = _bootstrap_ci(bs, "balanced_accuracy")
        f1_lo, f1_hi = _bootstrap_ci(bs, "f1_macro")
        granular.append({
            "task": TASK_LABEL.get(d["task"], d["task"]),
            "model": MODEL_LABEL.get(d["model"], d["model"]),
            "model_key": d["model"],
            "dataset": d["dataset"],
            "n_total": d["n_total"],
            "n_test": d["n_test"],
            "cv": "CV5" if d.get("cv_used") else "split",
            "BA": _fmt3(tm["balanced_accuracy"]),
            "BA_CI": f"[{_fmt3(ba_lo)}, {_fmt3(ba_hi)}]",
            "F1": _fmt3(tm["f1_macro"]),
            "F1_CI": f"[{_fmt3(f1_lo)}, {_fmt3(f1_hi)}]",
            "AUC": _fmt3(auc_v),
            "AUC_CI": f"[{_fmt3(auc_lo)}, {_fmt3(auc_hi)}]",
        })
    _write_csv(out_dir / "tabla6_utilidad_global.csv",
               ["task", "model", "dataset", "n_total", "n_test", "cv",
                "BA", "BA_CI", "F1", "F1_CI", "AUC", "AUC_CI"],
               granular)

    # ── Tabla 1b: utilidad global agregada (tarea × modelo) ─────────────────
    print("[2/4] Utilidad global agregada (tarea × modelo)")
    agregada = []
    for t in TASK_LABEL:
        for m in MODELS:
            rows = [r for r in granular if r["task"] == TASK_LABEL[t] and r["model_key"] == m]
            if not rows:
                continue
            def _mean(key):
                vals = [float(r[key]) for r in rows if r[key]]
                return float(np.mean(vals)) if vals else None
            agregada.append({
                "task": TASK_LABEL[t],
                "model": MODEL_LABEL[m],
                "model_key": m,
                "n_datasets": len(rows),
                "BA_agregada": _fmt3(_mean("BA")),
                "F1_agregada": _fmt3(_mean("F1")),
                "AUC_agregada": _fmt3(_mean("AUC")),
            })
    _write_csv(out_dir / "tabla6_utilidad_global_agregada.csv",
               ["task", "model", "n_datasets", "BA_agregada", "F1_agregada", "AUC_agregada"],
               agregada)

    # ── Tablas 2 y 3: fairness por sexo y edad ───────────────────────────────
    print("[3/4] Fairness por sexo y edad")
    sex_rows, age_rows = [], []
    for path in sorted(results_dir.glob("eval_*.json")):
        d = _load_json(path)
        for dim, target in (("sex", sex_rows), ("age", age_rows)):
            records = _fairness_rows(d, dim)
            if not records:
                continue
            row = _fairness_summary_row(d, dim, records)
            row["n_cover"] = sum(int(r["n"]) for r in records)
            target.append(row)
    _write_csv(out_dir / "tabla6_fairness_sex.csv",
               ["task", "model", "dataset", "n_test", "n_cover", "n_subgroups",
                "gap_BA", "gap_F1", "delta_AUC", "AUC_ES", "avg_gap",
                "tpr_gap", "fpr_gap", "best_auc_group", "worst_auc_group"],
               sex_rows)
    _write_csv(out_dir / "tabla6_fairness_age.csv",
               ["task", "model", "dataset", "n_test", "n_cover", "n_subgroups",
                "gap_BA", "gap_F1", "delta_AUC", "AUC_ES", "avg_gap",
                "tpr_gap", "fpr_gap", "tpr_best_group", "tpr_worst_group"],
               age_rows)

    # ── Tabla 4: cross-dataset ───────────────────────────────────────────────
    print("[4/4] Generalización cross-dataset (ID vs OOD)")
    cross_rows = []
    for path in sorted(results_dir.glob("cross_*_mlp.json")):
        d = _load_json(path)
        tr, te = d["train_dataset"], d["test_dataset"]
        id_metrics = id_index.get((d["model"], d["task"], tr))
        if id_metrics is None:
            print(f"  [aviso] sin ID para {d['model']}/{d['task']}/{tr}", file=sys.stderr)
            continue
        ood = d["metrics"]
        ba_id, ba_ood = id_metrics["balanced_accuracy"], ood.get("balanced_accuracy")
        f1_id, f1_ood = id_metrics["f1_macro"], ood.get("f1_macro")
        cross_rows.append({
            "task": TASK_LABEL.get(d["task"], d["task"]),
            "model_key": d["model"],
            "model": MODEL_LABEL.get(d["model"], d["model"]),
            "train": tr,
            "test": te,
            "n_train": d["n_train"],
            "n_test": d["n_test"],
            "BA_ID": _fmt3(ba_id),
            "BA_OOD": _fmt3(ba_ood),
            "caida_BA": _fmt3(ba_id - ba_ood if ba_id is not None and ba_ood is not None else None),
            "caida_BA_rel_pct": _fmt_pct(
                (ba_id - ba_ood) / ba_id * 100 if ba_id not in (None, 0) and ba_ood is not None else None
            ),
            "F1_ID": _fmt3(f1_id),
            "F1_OOD": _fmt3(f1_ood),
            "caida_F1": _fmt3(f1_id - f1_ood if f1_id is not None and f1_ood is not None else None),
            "caida_F1_rel_pct": _fmt_pct(
                (f1_id - f1_ood) / f1_id * 100 if f1_id not in (None, 0) and f1_ood is not None else None
            ),
            "AUC_OOD": _fmt3(_auc(ood)),
        })
    _write_csv(out_dir / "tabla6_cross_dataset.csv",
               ["task", "model", "train", "test", "n_train", "n_test",
                "BA_ID", "BA_OOD", "caida_BA", "caida_BA_rel_pct",
                "F1_ID", "F1_OOD", "caida_F1", "caida_F1_rel_pct", "AUC_OOD"],
               cross_rows)

    if args.figures:
        print("\nGenerando figuras...")
        figuras(agregada, cross_rows, out_dir)

    print(f"\nTodos los CSV exportados a: {out_dir}")


if __name__ == "__main__":
    main()
