#!/usr/bin/env python3
"""
Escrito/metricas/exportar_cross_delta_cis.py

Estima intervalos de confianza al 95% para la caída cross-dataset
(ΔBA y ΔF1) del Capítulo 6, sin re-ejecutar ninguna evaluación.

El bootstrap por paciente de (y_true, y_pred) es exactamente equivalente
a remuestrear multinomialmente las celdas de la matriz de confusión
agregada (los pacientes son iid), por lo que:

  1. Se reconstruyen las CMs ID y OOD desde los JSON guardados
     (eval_*.json y cross_*.json), tal como se presentan en el capítulo.
  2. Por iteración se remuestrea cada CM con su n_test fijo, se
     recalculan BA y macro-F1 con la misma semántica de sklearn usada
     en la evaluación original, y se guarda la diferencia
     ΔBA = BA_ID - BA_OOD (ídem ΔF1).
  3. El IC 95% es el intervalo percentil 2,5-97,5 de la distribución
     de la diferencia. Los conjuntos ID y OOD son disjuntos
     (pacientes distintos), por lo que remuestrear cada uno de forma
     independiente es correcto.

Validación: para cada escenario (eval y cross), compara el IC obtenido
remuestreando la CM contra el IC bootstrap ya guardado en el JSON (que
fue calculado remuestreando pacientes directamente). Si ambos coinciden
dentro del ruido de Monte Carlo, el procedimiento queda validado.

Salidas:
  - tabla6_cross_delta_cis.csv       ΔBA/ΔF1 con IC 95% por par train→test
  - tabla6_cross_delta_cis_validacion.csv  validación CM-bootstrap vs bootstrap original
  - tabla6_cross_dataset_con_ic.tex  tablas LaTeX actualizadas para 06-oe3.tex

Uso:
    python exportar_cross_delta_cis.py [--results-dir RUTA] [--output-dir RUTA]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)

N_BOOTSTRAP = 1000
SEED = 42
Z = 1.959963984540054

MODEL_LABEL = {
    "uni2": "UNI2",
    "virchow2": "Virchow2",
    "phikon_v2": "Phikon-v2",
    "prov_gigapath": "Prov-GigaPath",
}
MODEL_ORDER = ["UNI2", "Virchow2", "Phikon-v2", "Prov-GigaPath"]

TASK_LABEL = {
    "her2_status": "HER2",
    "molecular_subtype": "MolSub",
    "site": "Site",
    "side": "Side",
    "organ": "Organ",
}

TASK_NAME = {
    "HER2": "estado HER2",
    "MolSub": "subtipo molecular",
    "Site": "sitio anatómico",
    "Side": "lateralidad",
    "Organ": "clasificación de órgano",
}

DATASET_LABEL = {
    "BCNB": "BCNB",
    "HISTAI_Breast": "HISTAI-Breast",
    "HistologyHSI": "HSI-BC",
    "HISTAI_CRC_B1": "HISTAI-CRC-b1",
    "HISTAI_CRC_B2": "HISTAI-CRC-b2",
    "all": "Combinado (5 cohortes)",
}

TASKS_CROSS = ["HER2", "MolSub", "Site", "Side"]


def _find_results_dir() -> Path:
    env = os.environ.get("TESIS_RESULTS_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    here = Path(__file__).resolve().parent
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


# ─────────────────────────────────────────────────────────────────────────────
#  Métricas desde la matriz de confusión (misma semántica que compute_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _cm_to_arrays(cm: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruye y_true / y_pred a partir de la CM (fila = real, col = pred)."""
    cm = np.asarray(cm, dtype=int)
    ys_true, ys_pred = [], []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            n = int(cm[i, j])
            if n > 0:
                ys_true.extend([i] * n)
                ys_pred.extend([j] * n)
    return np.asarray(ys_true), np.asarray(ys_pred)


def _ba_f1_from_cm(cm: list[list[int]]) -> tuple[float | None, float | None]:
    """BA y macro-F1 desde una CM, replicando compute_metrics del repo.

    balanced_accuracy_score usa labels=unique(y_true); f1_score con
    average='macro' y labels=None usa la unión unique(y_true ∪ y_pred).
    Se reproduce exactamente pasándole las mismas llamadas de sklearn.
    """
    y_true, y_pred = _cm_to_arrays(cm)
    if len(np.unique(y_true)) < 2:
        return None, None
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(ba), float(f1)


def _bootstrap_metrics_from_cm(cm, n: int, rng: np.random.Generator) -> list[tuple]:
    """Remuestrea la CM n veces (multinomial) y devuelve [(ba, f1), ...]."""
    cm = np.asarray(cm, dtype=float)
    p = cm / cm.sum() if cm.sum() > 0 else cm
    flat = p.ravel()
    samples = []
    for _ in range(N_BOOTSTRAP):
        cells = rng.multinomial(n, flat).reshape(cm.shape)
        if (cells.sum(axis=1) > 0).sum() < 2:
            continue
        yt, yp = _cm_to_arrays(cells.astype(int).tolist())
        ba = balanced_accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        samples.append((float(ba), float(f1)))
    return samples


def _percentile_ci(vals: list[float], point: float | None) -> dict:
    if not vals:
        return {"point": point, "ci_low": None, "ci_high": None, "n_valid": 0}
    return {
        "point": point,
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_valid": len(vals),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap de la diferencia Δ = ID − OOD
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_delta(cm_id, n_id, cm_ood, n_ood) -> dict:
    """IC 95% de ΔBA y ΔF1 remuestreando ambas CMs de forma independiente.

    Los conjuntos ID y OOD son disjuntos (pacientes de datasets distintos),
    por lo que las muestras bootstrap de cada lado son independientes.
    """
    rng = np.random.default_rng(SEED)
    p_id = np.asarray(cm_id, dtype=float)
    p_id = p_id / p_id.sum() if p_id.sum() > 0 else p_id
    p_ood = np.asarray(cm_ood, dtype=float)
    p_ood = p_ood / p_ood.sum() if p_ood.sum() > 0 else p_ood

    d_ba, d_f1 = [], []
    for _ in range(N_BOOTSTRAP):
        cells_id = rng.multinomial(n_id, p_id.ravel()).reshape(p_id.shape)
        cells_ood = rng.multinomial(n_ood, p_ood.ravel()).reshape(p_ood.shape)
        if (cells_id.sum(axis=1) > 0).sum() < 2 or (cells_ood.sum(axis=1) > 0).sum() < 2:
            continue
        ba_id, f1_id = _ba_f1_from_cm(cells_id.astype(int).tolist())
        ba_ood, f1_ood = _ba_f1_from_cm(cells_ood.astype(int).tolist())
        if ba_id is None or ba_ood is None:
            continue
        d_ba.append(ba_id - ba_ood)
        d_f1.append(f1_id - f1_ood)

    ba_id_pt, f1_id_pt = _ba_f1_from_cm(cm_id)
    ba_ood_pt, f1_ood_pt = _ba_f1_from_cm(cm_ood)
    d_ba_pt = ba_id_pt - ba_ood_pt if ba_id_pt is not None and ba_ood_pt is not None else None
    d_f1_pt = f1_id_pt - f1_ood_pt if f1_id_pt is not None and f1_ood_pt is not None else None

    return {
        "ba_id": ba_id_pt,
        "ba_ood": ba_ood_pt,
        "delta_ba": _percentile_ci(d_ba, d_ba_pt),
        "delta_f1": _percentile_ci(d_f1, d_f1_pt),
        "delta_rel_pct": (
            (d_ba_pt / ba_id_pt * 100) if d_ba_pt is not None and ba_id_pt not in (None, 0) else None
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Validación: CM-bootstrap vs bootstrap de pacientes guardado
# ─────────────────────────────────────────────────────────────────────────────

def validar(results_dir: Path, eval_jsons: list[dict], cross_jsons: list[dict]) -> list[dict]:
    rows = []
    rng = np.random.default_rng(SEED + 1)

    def _check(tag: str, cm, stored_bs: dict):
        samples = _bootstrap_metrics_from_cm(cm, int(np.asarray(cm).sum()), rng)
        for metric, key in (("balanced_accuracy", "BA"), ("f1_macro", "F1")):
            entry = (stored_bs or {}).get(metric) or {}
            lo_s, hi_s = entry.get("ci_low"), entry.get("ci_high")
            if lo_s is None or hi_s is None:
                continue
            vals = [s[0] for s in samples] if key == "BA" else [s[1] for s in samples]
            if not vals:
                continue
            rows.append({
                "escenario": tag,
                "metrica": key,
                "ic_bootstrap_guardado": f"[{lo_s:.3f},{hi_s:.3f}]",
                "ic_cm_bootstrap": f"[{np.percentile(vals, 2.5):.3f},{np.percentile(vals, 97.5):.3f}]",
                "diff_lo": abs(np.percentile(vals, 2.5) - lo_s),
                "diff_hi": abs(np.percentile(vals, 97.5) - hi_s),
            })

    for d in eval_jsons:
        cm = d["probes"]["mlp"]["confusion_matrix"]["confusion_matrix"]
        _check(f"eval:{d['model']}:{d['task']}:{d['dataset']}", cm, d["probes"]["mlp"]["bootstrap"])

    for d in cross_jsons:
        cm = d["confusion_matrix"]["confusion_matrix"]
        _check(f"cross:{d['model']}:{d['task']}:{d['train_dataset']}->{d['test_dataset']}",
               cm, d.get("bootstrap"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def _fmt3(x) -> str:
    return "" if x is None else f"{float(x):.3f}"


def _fmt_ic(ci: dict) -> str:
    if ci["ci_low"] is None or ci["ci_high"] is None:
        return _fmt3(ci["point"])
    return f"{_fmt3(ci['point'])} [{_fmt3(ci['ci_low'])},{_fmt3(ci['ci_high'])}]"


def _fmt_rel(x) -> str:
    return "" if x is None else f"{float(x):.1f}"


def _ds_label(ds: str, cv: bool) -> str:
    lab = DATASET_LABEL.get(ds, ds)
    return f"{lab}*" if cv else lab


def _render_cross_table(out: list[str], task: str, caption: str, label: str,
                        rows: list[dict], n_cols: int = 9) -> None:
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\caption{" + caption + "}")
    out.append("\\label{" + label + "}")
    out.append("\\footnotesize")
    out.append("\\resizebox{\\textwidth}{!}{%")
    out.append("\\begin{tabular}{l l c c c c c c c}")
    out.append("\\toprule")
    out.append("\\textbf{Modelo} & \\textbf{Train $\\rightarrow$ Test} & "
               "\\textbf{n\\textsubscript{test}} & \\textbf{BA-ID} & \\textbf{BA-OOD} & "
               "\\textbf{$\\Delta$BA [IC 95\\%]} & \\textbf{$\\Delta$rel (\\%)} & "
               "\\textbf{Macro-F1 OOD} & \\textbf{$\\Delta$F1 [IC 95\\%]} \\\\")
    out.append("\\midrule")
    for i, r in enumerate(rows):
        if i == 0 or r["task"] != rows[i - 1]["task"]:
            if i > 0:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{{r['task']}}}}} \\\\")
            out.append("\\midrule")
        cells = [
            r["model"],
            f"{r['train']} $\\rightarrow$ {r['test']}",
            str(r["n_test"]),
            _fmt3(r["ba_id"]),
            _fmt3(r["ba_ood"]),
            _fmt_ic(r["delta_ba"]),
            _fmt_rel(r["delta_rel_pct"]),
            _fmt3(r["f1_ood"]),
            _fmt_ic(r["delta_f1"]),
        ]
        out.append(" & ".join(cells) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}}%")
    out.append("\\end{table}")
    out.append("")


def render_latex_cross(rows_by_task: dict) -> str:
    """Renderiza las tablas cross-dataset del capítulo con las columnas
    ΔBA [IC 95%] y ΔF1 [IC 95%] agregadas: una para HER2, una para MolSub
    y una combinada para Site/Side (como en el capítulo actual)."""
    caption_cross = (
        "Generalización cross-dataset. BA-ID corresponde a la balanced accuracy del "
        "conjunto de prueba del dataset de entrenamiento; BA-OOD al rendimiento al evaluar "
        "en el dataset objetivo. $\\Delta$BA y $\\Delta$F1 reportan la caída absoluta de cada "
        "métrica con su intervalo de confianza al 95\\% por bootstrap de la diferencia "
        "(percentiles 2,5 y 97,5; conjuntos ID y OOD independientes); la caída relativa se "
        "calcula sobre BA-ID. Los conjuntos marcados con * se evaluaron con validación "
        "cruzada de 5 pliegues por su tamaño reducido."
    )
    out = []
    for task, label, task_name in [
        ("HER2", "tab:resultados_cross_dataset_mama_her2", "estado HER2"),
        ("MolSub", "tab:resultados_cross_dataset_mama_molsub", "subtipo molecular"),
    ]:
        rows = rows_by_task.get(task, [])
        if not rows:
            continue
        caption = caption_cross.replace(
            "Generalización cross-dataset.",
            f"Generalización cross-dataset en la tarea de {task_name}.",
        )
        _render_cross_table(out, task, caption, label, rows)

    # Tabla combinada colorrectal (Site + Side)
    rows = rows_by_task.get("Site", []) + rows_by_task.get("Side", [])
    if rows:
        caption = (
            "Generalización cross-dataset en tareas colorrectales. BA-ID corresponde a la "
            "balanced accuracy del conjunto de prueba del dataset de entrenamiento; BA-OOD al "
            "rendimiento al evaluar en el dataset objetivo. $\\Delta$BA y $\\Delta$F1 reportan la "
            "caída absoluta de cada métrica con su intervalo de confianza al 95\\% por "
            "bootstrap de la diferencia (percentiles 2,5 y 97,5; conjuntos ID y OOD "
            "independientes); la caída relativa se calcula sobre BA-ID. Los conjuntos "
            "marcados con * se evaluaron con validación cruzada de 5 pliegues por su tamaño "
            "reducido."
        )
        _render_cross_table(out, "Site", caption, "tab:resultados_cross_dataset_colorrectal", rows)

    out.append(
        "\\par\\medskip\\noindent\\footnotesize \\textbf{Nota.} Cuando el conjunto de "
        "entrenamiento no contiene todas las clases del objetivo, la evaluación OOD se "
        "restringe a las clases compartidas. En HER2, HSI-BC* contiene solo dos de las "
        "tres clases, de modo que la BA-OOD de los pares con HSI-BC* como fuente se "
        "interpreta frente a un nivel de azar de 0,5 y no es directamente comparable con "
        "los demás pares.\n"
    )
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IC 95% de ΔBA y ΔF1 cross-dataset (Capítulo 6)")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _find_results_dir()
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_jsons = [_load_json(p) for p in sorted(results_dir.glob("eval_*.json"))]
    cross_jsons = [_load_json(p) for p in sorted(results_dir.glob("cross_*_mlp.json"))]
    print(f"  eval: {len(eval_jsons)} json | cross: {len(cross_jsons)} json")

    # Índice de CMs ID por (model, task, dataset)
    id_cm = {}
    id_cv = {}
    for d in eval_jsons:
        id_cm[(d["model"], d["task"], d["dataset"])] = d["probes"]["mlp"]["confusion_matrix"]["confusion_matrix"]
        id_cv[(d["model"], d["task"], d["dataset"])] = bool(d.get("cv_used"))

    rows_by_task = {t: [] for t in TASKS_CROSS}
    for d in cross_jsons:
        key_id = (d["model"], d["task"], d["train_dataset"])
        if key_id not in id_cm:
            print(f"  [aviso] sin CM ID para {key_id}", file=sys.stderr)
            continue
        cm_id = id_cm[key_id]
        n_id = int(np.asarray(cm_id).sum())
        cm_ood = d["confusion_matrix"]["confusion_matrix"]
        n_ood = int(np.asarray(cm_ood).sum())

        res = bootstrap_delta(cm_id, n_id, cm_ood, n_ood)
        task = TASK_LABEL.get(d["task"], d["task"])
        cv_test = id_cv.get((d["model"], d["task"], d["test_dataset"]), False)
        rows_by_task[task].append({
            "task": task,
            "model": MODEL_LABEL.get(d["model"], d["model"]),
            "train": _ds_label(d["train_dataset"], id_cv.get(key_id, False)),
            "test": _ds_label(d["test_dataset"], cv_test),
            "n_train": d["n_train"],
            "n_test": d["n_test"],
            "ba_id": res["ba_id"],
            "ba_ood": res["ba_ood"],
            "f1_ood": d["metrics"].get("f1_macro"),
            "delta_ba": res["delta_ba"],
            "delta_f1": res["delta_f1"],
            "delta_rel_pct": res["delta_rel_pct"],
        })

    # CSV
    import csv

    csv_path = out_dir / "tabla6_cross_delta_cis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "train", "test", "n_test",
                    "BA_ID", "BA_OOD", "delta_BA", "delta_BA_ci_low", "delta_BA_ci_high",
                    "delta_F1", "delta_F1_ci_low", "delta_F1_ci_high", "delta_rel_pct"])
        for t in TASKS_CROSS:
            for r in rows_by_task[t]:
                w.writerow([
                    r["task"], r["model"], r["train"], r["test"], r["n_test"],
                    _fmt3(r["ba_id"]), _fmt3(r["ba_ood"]),
                    _fmt3(r["delta_ba"]["point"]), _fmt3(r["delta_ba"]["ci_low"]), _fmt3(r["delta_ba"]["ci_high"]),
                    _fmt3(r["delta_f1"]["point"]), _fmt3(r["delta_f1"]["ci_low"]), _fmt3(r["delta_f1"]["ci_high"]),
                    _fmt3(r["delta_rel_pct"]),
                ])
    print(f"  -> {csv_path}")

    # Validación
    print("\nValidando CM-bootstrap contra bootstrap de pacientes guardado...")
    val = validar(results_dir, eval_jsons, cross_jsons)
    diffs_lo = [r["diff_lo"] for r in val]
    diffs_hi = [r["diff_hi"] for r in val]
    if diffs_lo:
        print(f"  n={len(val)} | diff percentil 2.5: mediana={np.median(diffs_lo):.4f} "
              f"máx={np.max(diffs_lo):.4f} | diff percentil 97.5: mediana={np.median(diffs_hi):.4f} "
              f"máx={np.max(diffs_hi):.4f}")
    val_path = out_dir / "tabla6_cross_delta_cis_validacion.csv"
    with open(val_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["escenario", "metrica", "ic_bootstrap_guardado",
                                          "ic_cm_bootstrap", "diff_lo", "diff_hi"])
        w.writeheader()
        w.writerows(val)
    print(f"  -> {val_path}")

    # LaTeX
    tex_path = out_dir / "tabla6_cross_dataset_con_ic.tex"
    tex_path.write_text(render_latex_cross(rows_by_task), encoding="utf-8")
    print(f"  -> {tex_path}")


if __name__ == "__main__":
    main()
