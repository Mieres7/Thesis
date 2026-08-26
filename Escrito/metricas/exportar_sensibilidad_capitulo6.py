#!/usr/bin/env python3
"""
Escrito/metricas/exportar_sensibilidad_capitulo6.py

Exporta la sensibilidad (recall) por clase del Capítulo 6, calculada
desde los JSON generados por eval/sandbox/run_eval.py.

La sensibilidad por clase ya está almacenada en cada eval_*.json bajo
`probes.mlp.test_metrics.per_class_sensitivity` (punto estimado). Los
intervalos de confianza al 95% se calculan aquí con el intervalo exacto
de Wilson sobre la matriz de confusión ya persistida
(`probes.mlp.confusion_matrix.per_class.<cls>.tp/.fn`), de modo que el
punto estimado coincide exactamente con lo reportado sin reentrenar nada.

Nota metodológica: la balanced accuracy ya es la media del recall por
clase, por lo que esta tabla aporta el desglose por clase que la BA
promedia y que no era visible en las tablas de utilidad global.

Salidas (en el mismo directorio de este script):
  - tabla6_sensibilidad_por_clase.csv   (punto + IC 95% Wilson por clase)
  - tabla6_sensibilidad_por_clase.tex   (tablas LaTeX listas para 06-oe3.tex)

Uso:
    python exportar_sensibilidad_capitulo6.py [--results-dir RUTA]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

MODEL_LABEL = {
    "uni2": "UNI2",
    "virchow2": "Virchow2",
    "phikon_v2": "Phikon-v2",
    "prov_gigapath": "Prov-GigaPath",
}

TASK_LABEL = {
    "her2_status": "HER2",
    "molecular_subtype": "MolSub",
    "site": "Site",
    "side": "Side",
    "organ": "Organ",
}

# Orden de presentación de las tareas (grupos dentro de cada tabla)
TASK_ORDER = ["HER2", "MolSub", "Site", "Side", "Organ"]

# Nombres cortos de clase para las cabeceras LaTeX (idem a los del JSON)
CLASS_SHORT = {
    "colon_unspecified": "colon unspec.",
    "left_colon": "colon izq.",
    "metastatic_liver": "met. hígado",
    "other": "otro",
    "rectosigmoid": "rectosigmoide",
    "rectum": "recto",
    "right_colon": "colon der.",
    "sigmoid": "sigmoide",
    "transverse_colon": "colon transv.",
    "left": "izquierdo",
    "right": "derecho",
    "transverse": "transverso",
    "breast": "mama",
    "colorectal": "colorrectal",
}

# Claves de clase por tarea en el orden natural del JSON
TASK_CLASS_ORDER = {
    "HER2": None,  # se toma del primer JSON de la tarea
    "MolSub": None,
    "Site": None,
    "Side": None,
    "Organ": None,
}

# Claves de dataset por tarea para ordenar filas (train principal → externos)
DATASET_ORDER = {
    "HER2": ["BCNB", "HISTAI_Breast", "HistologyHSI"],
    "MolSub": ["BCNB", "HISTAI_Breast", "HistologyHSI"],
    "Site": ["HISTAI_CRC_B1", "HISTAI_CRC_B2"],
    "Side": ["HISTAI_CRC_B1", "HISTAI_CRC_B2"],
    "Organ": ["all"],
}

DATASET_LABEL = {
    "BCNB": "BCNB",
    "HISTAI_Breast": "HISTAI-Breast",
    "HistologyHSI": "Histology*",
    "HISTAI_CRC_B1": "HISTAI-CRC-b1",
    "HISTAI_CRC_B2": "HISTAI-CRC-b2*",
    "all": "Combinado (5 cohortes)",
}


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


def wilson_ci(tp: int, fn: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Intervalo de Wilson al 95% para una proporción p = tp/(tp+fn)."""
    n = tp + fn
    if n <= 0:
        return 0.0, 0.0
    p = tp / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) + z2 / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _fmt3(x: float) -> str:
    return f"{x:.3f}"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{_fmt3(lo)},{_fmt3(hi)}]"


def _short_label(cls) -> str:
    s = CLASS_SHORT.get(str(cls))
    if s is not None:
        return s
    try:
        v = float(cls)
        if v == int(v):
            return str(int(v))
    except (TypeError, ValueError):
        pass
    return str(cls)


def _per_class_counts(d: dict) -> dict:
    """Devuelve {str(indice): (tp, fn)} desde la confusion_matrix.

    Las claves de `per_class` son los labels del JSON (str del índice de
    clase), por lo que indexar por índice es lo correcto.
    """
    cm = d["probes"]["mlp"]["confusion_matrix"]
    per_class = cm["per_class"]
    n_classes = len(cm["labels"])
    out = {}
    for i in range(n_classes):
        pc = per_class.get(str(i)) or per_class.get(i)
        if pc is None:
            continue
        out[str(i)] = (int(pc["tp"]), int(pc["fn"]))
    return out


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("eval_*.json")):
        d = _load_json(path)
        tm = d["probes"]["mlp"]["test_metrics"]
        sens = tm.get("per_class_sensitivity") or {}
        counts = _per_class_counts(d)
        task = TASK_LABEL.get(d["task"], d["task"])
        for key, val in sens.items():
            if not key.startswith("class_"):
                continue
            idx = int(key.split("_")[1])
            cls_orig = d["classes"][idx] if idx < len(d["classes"]) else key
            tp, fn = counts.get(str(idx), (0, 0))
            lo, hi = wilson_ci(tp, fn)
            rows.append({
                "task": task,
                "model": MODEL_LABEL.get(d["model"], d["model"]),
                "model_key": d["model"],
                "dataset": d["dataset"],
                "dataset_label": DATASET_LABEL.get(d["dataset"], d["dataset"]),
                "cv": "CV5" if d.get("cv_used") else "split",
                "n_test": d["n_test"],
                "class": str(cls_orig),
                "class_label": _short_label(cls_orig),
                "sensitivity": _fmt3(val),
                "ci_low": _fmt3(lo),
                "ci_high": _fmt3(hi),
                "ci": _fmt_ci(lo, hi),
                "tp": tp,
                "fn": fn,
            })
    return rows


def _class_order(rows: list[dict], task: str) -> list[str]:
    """Clases en el orden natural de aparición (índice de clase)."""
    seen = []
    for r in rows:
        if r["task"] != task:
            continue
        cls = r["class"]
        if cls not in seen:
            seen.append(cls)
    return seen


def render_latex(rows: list[dict]) -> str:
    out = []
    for task in TASK_ORDER:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        classes = _class_order(task_rows, task)
        n_classes = len(classes)

        # Orden de filas: dataset (según DATASET_ORDER) × modelo (UNI2...Prov)
        ds_order = [d for d in DATASET_ORDER[task] if any(r["dataset"] == d for r in task_rows)]
        model_order = ["UNI2", "Virchow2", "Phikon-v2", "Prov-GigaPath"]
        ordered = []
        for ds in ds_order:
            for mdl in model_order:
                match = next((r for r in task_rows
                              if r["dataset"] == ds and r["model"] == mdl), None)
                if match is not None:
                    ordered.append(match)

        n_cols = 3 + n_classes  # Conjunto | Modelo | n_test | clases...
        task_name = {
            "HER2": "estado HER2",
            "MolSub": "subtipo molecular",
            "Site": "sitio anatómico",
            "Side": "lateralidad",
            "Organ": "clasificación de órgano",
        }[task]
        ref_tab = ("tab:resultados_utilidad_crc"
                   if task in ("Site", "Side") else "tab:resultados_utilidad_mama")
        caption = (
            f"Sensibilidad (recall) por clase en la tarea de {task_name}. "
            f"Se reporta el recall de cada clase sobre el conjunto de prueba con su "
            f"intervalo de confianza al 95\\% exacto (Wilson), calculado a partir de "
            f"la matriz de confusión. La balanced accuracy reportada en la "
            f"Tabla~\\ref{{{ref_tab}}} corresponde a la media de "
            f"estos valores por clase."
        )
        label = f"tab:sensibilidad_{task.lower()}"

        out.append("\\begin{table}[htbp]")
        out.append("\\centering")
        out.append("\\caption{" + caption + "}")
        out.append("\\label{" + label + "}")
        out.append("\\footnotesize")
        out.append("\\resizebox{\\textwidth}{!}{%")
        col_spec = " ".join(["l", "l", "c"] + ["c"] * n_classes)
        out.append("\\begin{tabular}{" + col_spec + "}")
        out.append("\\toprule")
        header = ["\\textbf{Conjunto}", "\\textbf{Modelo}", "\\textbf{n\\textsubscript{test}}"]
        for cls in classes:
            header.append("\\textbf{Sens. " + _short_label(cls) + "}")
        out.append(" & ".join(header) + " \\\\")
        out.append("\\midrule")

        # Agrupar por dataset
        prev_ds = None
        for r in ordered:
            if prev_ds is not None and r["dataset"] != prev_ds:
                out.append("\\midrule")
            prev_ds = r["dataset"]
            cells = [r["dataset_label"], r["model"], str(r["n_test"])]
            lookup = {x["class"]: x for x in task_rows
                      if x["dataset"] == r["dataset"] and x["model"] == r["model"]}
            for cls in classes:
                item = lookup.get(cls)
                cells.append(f"{item['sensitivity']} {item['ci']}" if item else "--")
            out.append(" & ".join(cells) + " \\\\")

        out.append("\\bottomrule")
        out.append("\\end{tabular}}%")
        out.append("\\end{table}")
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Exporta sensibilidad por clase del Capítulo 6")
    parser.add_argument("--results-dir", default=None, help="Directorio con los JSON de resultados")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida (default: este script)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _find_results_dir()
    if not results_dir.is_dir():
        raise SystemExit(f"No existe el directorio de resultados: {results_dir}")
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(results_dir)
    if not rows:
        raise SystemExit("No se encontraron filas de sensibilidad por clase")

    csv_path = out_dir / "tabla6_sensibilidad_por_clase.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "task", "model", "dataset", "dataset_label", "cv", "n_test",
            "class", "class_label", "sensitivity", "ci_low", "ci_high", "ci",
            "tp", "fn",
        ], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {csv_path} ({len(rows)} filas)")

    tex_path = out_dir / "tabla6_sensibilidad_por_clase.tex"
    tex_path.write_text(render_latex(rows), encoding="utf-8")
    print(f"  -> {tex_path}")


if __name__ == "__main__":
    main()
