#!/usr/bin/env python3
"""
Escrito/resultados_cap6/generar_tablas_anexo.py

Genera las tablas del ANEXO de cap6-restructura.tex (resultados completos)
directamente desde los CSVs exportados (misma fuente que las tablas del
capítulo; no recalcula nada).

Uso:
    python generar_tablas_anexo.py --task HER2   # una tarea
    python generar_tablas_anexo.py --all         # todas las tareas

Salidas (en este directorio):
    anexo_<tarea>.tex   -> 4 bloques LaTeX por tarea (utilidad, sensibilidad,
                           fairness, cross-dataset) listos para pegar en el
                           capítulo, con los labels del esqueleto
                           (tab:anexo_<tarea>_*).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_V2 = Path(r"C:\Users\vicen\Repositories\tesis\eval\sandbox\results_v2")

DATASET_LABEL = {
    "BCNB": "BCNB",
    "HISTAI_Breast": "HISTAI-Breast",
    "HistologyHSI": "HSI-BC",
    "HISTAI_CRC_B1": "HISTAI-CRC-B1",
    "HISTAI_CRC_B2": "HISTAI-CRC-B2",
    "SurGen": "SurGen",
    "all": "Combinado",
    "HISTAI-Breast": "HISTAI-Breast",
    "HSI-BC": "HSI-BC",
    "HISTAI-CRC-b1": "HISTAI-CRC-B1",
    "HISTAI-CRC-b2": "HISTAI-CRC-B2",
}

MODEL_ORDER = ["UNI2", "Virchow2", "Phikon-v2", "Prov-GigaPath"]

# Config por tarea: datasets, clases [(clave_csv, etiqueta)], fairness por
# edad/sexo, pares cross (claves CSV sin '*')
TASKS = {
    "HER2": {
        "codigo": "T1-HER2",
        "label_anexo": "her2",
        "datasets": ["BCNB", "HISTAI_Breast", "HistologyHSI"],
        "clases": [("0.0", "0/1+"), ("1.0", "2+"), ("2.0", "3+")],
        "fairness_dims": ["age"],
        "cross_pairs": [
            ("BCNB", "HISTAI-Breast"), ("BCNB", "HSI-BC"),
            ("HISTAI-Breast", "BCNB"), ("HISTAI-Breast", "HSI-BC"),
            ("HSI-BC", "BCNB"), ("HSI-BC", "HISTAI-Breast"),
        ],
    },
    "MolSub": {
        "codigo": "T2-MolSub",
        "label_anexo": "molsub",
        "datasets": ["BCNB", "HISTAI_Breast", "HistologyHSI"],
        "clases": [("Luminal A", "Luminal A"), ("Luminal B", "Luminal B"),
                   ("HER2+", "HER2+"), ("Triple Negative", "Triple Negative")],
        "fairness_dims": ["age"],
        "cross_pairs": [
            ("BCNB", "HISTAI-Breast"), ("BCNB", "HSI-BC"),
            ("HISTAI-Breast", "BCNB"), ("HISTAI-Breast", "HSI-BC"),
            ("HSI-BC", "BCNB"), ("HSI-BC", "HISTAI-Breast"),
        ],
    },
    "Site": {
        "codigo": "T3-Site",
        "label_anexo": "site",
        "datasets": ["HISTAI_CRC_B1", "HISTAI_CRC_B2", "SurGen"],
        "clases": [("right_colon", "colon der."), ("left_colon", "colon izq."),
                   ("sigmoid", "sigmoide"), ("transverse_colon", "colon transv."),
                   ("colon_unspecified", "colon unspec."), ("rectosigmoid", "rectosigmoide"),
                   ("rectum", "recto"), ("metastatic_liver", "met. hígado"),
                   ("other", "otro")],
        "fairness_dims": ["age", "sex"],
        "sens_transpuesta": True,
        "cross_pairs": [
            ("HISTAI-CRC-b1", "HISTAI-CRC-b2"), ("HISTAI-CRC-b1", "SurGen"),
            ("HISTAI-CRC-b2", "HISTAI-CRC-b1"), ("HISTAI-CRC-b2", "SurGen"),
            ("SurGen", "HISTAI-CRC-b1"), ("SurGen", "HISTAI-CRC-b2"),
        ],
    },
    "Side": {
        "codigo": "T4-Side",
        "label_anexo": "side",
        "datasets": ["HISTAI_CRC_B1", "HISTAI_CRC_B2", "SurGen"],
        "clases": [("left", "izquierdo"), ("right", "derecho"),
                   ("transverse", "transverso"), ("rectosigmoid", "rectosigmoide"),
                   ("rectum", "recto")],
        "fairness_dims": ["age", "sex"],
        "cross_pairs": [
            ("HISTAI-CRC-b1", "HISTAI-CRC-b2"), ("HISTAI-CRC-b1", "SurGen"),
            ("HISTAI-CRC-b2", "HISTAI-CRC-b1"), ("HISTAI-CRC-b2", "SurGen"),
            ("SurGen", "HISTAI-CRC-b1"), ("SurGen", "HISTAI-CRC-b2"),
        ],
    },
    "Organ": {
        "codigo": "T5-Organ",
        "label_anexo": "organ",
        "datasets": ["all"],
        "clases": [("breast", "mama"), ("colorectal", "colorrectal")],
        "fairness_dims": ["age"],
        "cross_pairs": [],
    },
}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt3(x) -> str:
    if x in (None, ""):
        return "--"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_ic(lo, hi) -> str:
    if lo in (None, "") or hi in (None, ""):
        return "--"
    return f"[{float(lo):.3f}, {float(hi):.3f}]"


def _fmt1(x) -> str:
    if x in (None, ""):
        return "--"
    try:
        return f"{float(x):.1f}"
    except (TypeError, ValueError):
        return str(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Utilidad global
# ─────────────────────────────────────────────────────────────────────────────

def tabla_utilidad(task: str, cfg: dict) -> list[str]:
    rows = _read_csv(HERE / "tabla6_utilidad_global.csv")
    out = [rf"\begin{{table}}[H]", r"\centering",
           rf"\caption{{Utilidad global por modelo, tarea {cfg["codigo"]}. IC 95\% obtenido por bootstrap (1000 remuestreos, a nivel de paciente).}}",
           rf"\label{{tab:anexo_{cfg['label_anexo']}_utilidad}}",
           r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{l l c c c c c c}", r"\toprule",
           r"\textbf{Dataset} & \textbf{Modelo} & \textbf{n\_test} & \textbf{BA} & "
           r"\textbf{IC95\% BA} & \textbf{Macro-F1} & \textbf{IC95\% F1} & \textbf{AUC (OVR macro)} \\",
           r"\midrule"]
    for ds in cfg["datasets"]:
        for m in MODEL_ORDER:
            r = next((x for x in rows if x["task"] == task and x["dataset"] == ds and x["model"] == m), None)
            if r is None:
                continue
            out.append(f"{DATASET_LABEL[ds]} & {m} & {r['n_test']} & {_fmt3(r['BA'])} & "
                       f"{_fmt_ic(r['BA_CI'].strip('[]').split(',')[0], r['BA_CI'].strip('[]').split(',')[1])} & "
                       f"{_fmt3(r['F1'])} & "
                       f"{_fmt_ic(r['F1_CI'].strip('[]').split(',')[0], r['F1_CI'].strip('[]').split(',')[1])} & "
                       f"{_fmt3(r['AUC'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sensibilidad por clase (una tabla por dataset: clases en filas, modelos en
#    columnas; caben a ancho de página sin escalar)
# ─────────────────────────────────────────────────────────────────────────────

SENS_LABEL_DS = {
    "BCNB": "bcnb",
    "HISTAI_Breast": "histai",
    "HistologyHSI": "hsibc",
    "HISTAI_CRC_B1": "b1",
    "HISTAI_CRC_B2": "b2",
    "SurGen": "surgen",
    "all": "all",
}


def tabla_sensibilidad(task: str, cfg: dict) -> list[str]:
    rows = _read_csv(HERE / "tabla6_sensibilidad_por_clase.csv")
    clases = cfg["clases"]
    n_clases = len(clases)
    by_key = {(x["task"], x["dataset"], x["model"], x["class"]): x for x in rows}

    def sens(ds, m, key):
        x = by_key.get((task, ds, m, key))
        if x is None or x["sensitivity"] in ("", "--"):
            return "--"
        return _fmt3(x["sensitivity"])

    out: list[str] = []
    for idx, ds in enumerate(cfg["datasets"]):
        label = (f"tab:anexo_{cfg['label_anexo']}_sensibilidad" if idx == 0
                 else f"tab:anexo_{cfg['label_anexo']}_sensibilidad_{SENS_LABEL_DS[ds]}")
        out += [rf"\begin{{table}}[H]", r"\centering",
                rf"\caption{{Sensibilidad (recall) por clase, tarea {cfg["codigo"]}, dataset {DATASET_LABEL[ds]} ($K={n_clases}$). Clases en filas y un modelo por columna; las clases ausentes en la cohorte se marcan con ``--''.}}",
                rf"\label{{{label}}}",
                r"\begin{tabular}{l c c c c}", r"\toprule",
                r"\textbf{Clase} & \textbf{UNI2} & \textbf{Virchow2} & \textbf{Phikon-v2} & \textbf{Prov-GigaPath} \\",
                r"\midrule"]
        nrow = [r"\textbf{n\_test}"]
        for m in MODEL_ORDER:
            sub = next((x for x in rows if x["task"] == task and x["dataset"] == ds and x["model"] == m), None)
            nrow.append(sub["n_test"] if sub else "--")
        out.append(" & ".join(nrow) + r" \\")
        for key, lab in clases:
            vals = [sens(ds, m, key) for m in MODEL_ORDER]
            out.append(f"{lab} & " + " & ".join(vals) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fairness por subgrupos (edad y sexo si aplica)
# ─────────────────────────────────────────────────────────────────────────────

def _read_fairness_subgroups(task_key: str, ds: str, model: str, dim: str) -> list[dict]:
    model_key = {"UNI2": "uni2", "Virchow2": "virchow2", "Phikon-v2": "phikon_v2",
                 "Prov-GigaPath": "prov_gigapath"}[model]
    ds_key = {"BCNB": "BCNB", "HISTAI_Breast": "HISTAI_Breast", "HistologyHSI": "HistologyHSI",
              "HISTAI_CRC_B1": "HISTAI_CRC_B1", "HISTAI_CRC_B2": "HISTAI_CRC_B2",
              "SurGen": "SurGen", "all": "all"}[ds]
    task_key = {"HER2": "breast_her2", "MolSub": "breast_molsub",
                "Site": "crc_site", "Side": "crc_side", "Organ": "organ"}[task_key]
    path = RESULTS_V2 / f"fairness_{model_key}_{task_key}_{ds_key}_mlp_{dim}.csv"
    if not path.exists():
        return []
    return _read_csv(path)


def tabla_fairness(task: str, cfg: dict) -> list[str]:
    out: list[str] = []
    dims = cfg.get("fairness_dims", ["age"])
    for dim in dims:
        dim_label = "edad" if dim == "age" else "sexo"
        dim_tex = "de edad" if dim == "age" else "de sexo"
        summary_file = HERE / ("tabla6_fairness_age.csv" if dim == "age" else "tabla6_fairness_sex.csv")
        summary_rows = _read_csv(summary_file)
        if dim == "age":
            order = {"<50": 0, "50-70": 1, ">=70": 2}
            group_show = lambda g: g["group"]
        else:
            order = {"f": 0, "m": 1}
            group_show = lambda g: g["group"].upper()
        label_base = (f"tab:anexo_{cfg['label_anexo']}_fairness"
                      if len(dims) == 1
                      else f"tab:anexo_{cfg['label_anexo']}_fairness_{dim_label}")
        for idx, ds in enumerate(cfg["datasets"]):
            label_fairness = label_base if idx == 0 else f"{label_base}_{SENS_LABEL_DS[ds]}"
            out += [rf"\begin{{table}}[H]", r"\centering",
                    rf"\caption{{Fairness por subgrupo {dim_tex}, tarea {cfg["codigo"]}, dataset {DATASET_LABEL[ds]}. Solo se incluyen celdas con $n\geq 10$; el resto se marca como excluida. En tareas multiclase el TPR por subgrupo es la media del recall por clase (equivalente a la balanced accuracy cuando todas las clases están presentes en el subgrupo). $\Delta$BA y $\Delta$AUC corresponden a la brecha máxima entre subgrupos por combinación dataset-modelo.}}",
                    rf"\label{{{label_fairness}}}",
                    r"\resizebox{\textwidth}{!}{%",
                    r"\begin{tabular}{l l l c c c c c}", r"\toprule",
                    r"\textbf{Dataset} & \textbf{Modelo} & \textbf{Subgrupo} & \textbf{n} & "
                    r"\textbf{BA} & \textbf{TPR} & \textbf{FPR} & \textbf{$\Delta$BA / $\Delta$AUC} \\",
                    r"\midrule"]
            for m in MODEL_ORDER:
                sub = _read_fairness_subgroups(task, ds, m, dim)
                if not sub:
                    continue
                sub = sorted(sub, key=lambda x: order.get(x["group"], 9))
                summary = next((x for x in summary_rows if x["task"] == task and x["dataset"] == ds and x["model"] == m), None)
                n_groups = len(sub)
                for i, g in enumerate(sub):
                    has_metrics = "balanced_accuracy" in g and g["balanced_accuracy"] not in ("", None)
                    if g.get("low_n") == "True" or int(g["n"]) < 10 or not has_metrics:
                        out.append(f"{DATASET_LABEL[ds]} & {m} & {group_show(g)} (excluida) & {g['n']} & "
                                   "-- & -- & -- & \\\\")
                        continue
                    delta = ""
                    if summary is not None and i == n_groups - 1:
                        delta = (rf"$\Delta$BA={_fmt3(summary['gap_BA'])}; "
                                 rf"$\Delta$AUC={_fmt3(summary['delta_AUC'])}")
                    out.append(f"{DATASET_LABEL[ds]} & {m} & {group_show(g)} & {g['n']} & "
                               f"{_fmt3(g['balanced_accuracy'])} & {_fmt3(g['tpr'])} & "
                               f"{_fmt3(g['fpr'])} & {delta} \\\\")
                out.append(r"\midrule")
            out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-dataset (ID vs OOD)
# ─────────────────────────────────────────────────────────────────────────────

def tabla_cross(task: str, cfg: dict) -> list[str]:
    if not cfg["cross_pairs"]:
        return []
    rows = _read_csv(HERE / "tabla6_cross_delta_cis.csv")
    out = [rf"\begin{{table}}[H]", r"\centering",
           rf"\caption{{Generalización cross-dataset (ID vs. OOD), tarea {cfg["codigo"]}. "
            r"$\Delta_{\text{OOD}} = BA_{ID} - BA_{OOD}$; $\Delta_{\text{rel}}$ en \%.}",
           rf"\label{{tab:anexo_{cfg['label_anexo']}_crossdataset}}",
           r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{l l l c c c c c}", r"\toprule",
           r"\textbf{Train (ID)} & \textbf{Test (OOD)} & \textbf{Modelo} & "
           r"\textbf{BA ID} & \textbf{BA OOD} & \textbf{$\Delta_{\text{OOD}}$} & "
           r"\textbf{IC95\% $\Delta$} & \textbf{$\Delta_{\text{rel}}$ (\%)} \\",
           r"\midrule"]
    for tr, te in cfg["cross_pairs"]:
        for m in MODEL_ORDER:
            r = next((x for x in rows if x["task"] == task and x["model"] == m
                      and x["train"].rstrip("*") == tr and x["test"].rstrip("*") == te), None)
            if r is None:
                continue
            out.append(f"{DATASET_LABEL[tr]} & {DATASET_LABEL[te]} & {m} & "
                       f"{_fmt3(r['BA_ID'])} & {_fmt3(r['BA_OOD'])} & {_fmt3(r['delta_BA'])} & "
                       f"{_fmt_ic(r['delta_BA_ci_low'], r['delta_BA_ci_high'])} & "
                       f"{_fmt1(r['delta_rel_pct'])} \\\\")
        out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS) + ["ALL"])
    args = ap.parse_args()

    tasks = list(TASKS) if args.task == "ALL" else [args.task]
    for task in tasks:
        cfg = TASKS[task]
        bloque = (tabla_utilidad(task, cfg) + tabla_sensibilidad(task, cfg)
                  + tabla_fairness(task, cfg) + tabla_cross(task, cfg))
        label = cfg["label_anexo"]
        out_path = HERE / f"anexo_{label}.tex"
        out_path.write_text("\n".join(bloque) + "\n", encoding="utf-8")
        print(f"  -> {out_path.name} ({len(bloque)} líneas)")


if __name__ == "__main__":
    main()