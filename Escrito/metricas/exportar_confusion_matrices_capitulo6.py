#!/usr/bin/env python3
"""
Escrito/metricas/exportar_confusion_matrices_capitulo6.py

Genera las figuras de matrices de confusión normalizadas (por fila,
escala recall) del Capítulo 6, para todas las tareas y escenarios,
directamente desde los JSON guardados por eval/sandbox/run_eval.py.

Escenarios:
  - Intra-dataset (eval_*.json): HER2, MolSub, Site, Side y Organ,
    para los cuatro modelos fundacionales.
  - Cross-dataset (cross_*_mlp.json): Site y Side en ambas direcciones
    (HISTAI-CRC-b1 <-> HISTAI-CRC-b2), para los cuatro modelos.

Normalización: cada celda (fila = clase real, columna = clase predicha)
se divide por el total de la fila; la diagonal es, por lo tanto, el
recall de cada clase (misma escala que las tablas de sensibilidad).
Las celdas muestran además el conteo absoluto.

Estilo: replicado de Escrito/metricas/bias_analysis (rcParams, dpi 200).

Salidas (en <output-dir>, default Escrito/images):
  - fig6_cm_her2.png, fig6_cm_molsub.png, fig6_cm_organ.png
  - fig6_cm_site_b1.png, fig6_cm_site_b2.png, fig6_cm_side.png
  - fig6_cm_site_cross_b1_to_b2.png, fig6_cm_site_cross_b2_to_b1.png
  - fig6_cm_side_cross.png

Además imprime los pares de clases más confundidos por escenario, útiles
para el texto de análisis del capítulo.

Uso:
    python exportar_confusion_matrices_capitulo6.py [--results-dir RUTA] [--output-dir RUTA]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["uni2", "virchow2", "phikon_v2", "prov_gigapath"]
MODEL_LABEL = {
    "uni2": "UNI2",
    "virchow2": "Virchow2",
    "phikon_v2": "Phikon-v2",
    "prov_gigapath": "Prov-GigaPath",
}

DATASET_LABEL = {
    "BCNB": "BCNB",
    "HISTAI_Breast": "HISTAI-Breast",
    "HistologyHSI": "HSI-BC*",
    "HISTAI_CRC_B1": "HISTAI-CRC-b1",
    "HISTAI_CRC_B2": "HISTAI-CRC-b2*",
    "all": "Combinado (5 cohortes)",
}

CLASS_SHORT = {
    "colon_unspecified": "colon\nunspec.",
    "left_colon": "colon\nizq.",
    "metastatic_liver": "met.\nhígado",
    "other": "otro",
    "rectosigmoid": "recto-\nsigmoide",
    "rectum": "recto",
    "right_colon": "colon\nder.",
    "sigmoid": "sigmoide",
    "transverse_colon": "colon\ntransv.",
    "left": "izquierdo",
    "right": "derecho",
    "transverse": "transverso",
    "breast": "mama",
    "colorectal": "colorrectal",
}

# Nombres para las tareas de mama (coinciden con las tablas del capítulo)
CLASS_SPECIAL = {
    "her2_status": {
        "0.0": "0/1+",
        "1.0": "2+",
        "2.0": "3+",
    },
    "molecular_subtype": {
        "HER2(+)": "HER2+",
        "Triple negative": "Triple\nNegative",
    },
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
})


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


def _class_label(task: str, cls) -> str:
    s = str(cls)
    special = CLASS_SPECIAL.get(task, {})
    if s in special:
        return special[s]
    lab = CLASS_SHORT.get(s)
    if lab is not None:
        return lab
    return s


def _find_fontsize(n_classes: int, annotate: bool) -> tuple[float, float]:
    """(tick_label_size, cell_text_size) según la cantidad de clases."""
    if n_classes <= 3:
        return 11, 9
    if n_classes == 4:
        return 10, 8
    if n_classes == 5:
        return 9, 7
    return 6.5, 5.5


def _plot_cm(ax, cm: list[list[int]], classes: list[str], task: str, title: str,
             annotate: bool = True):
    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    row_sums = cm.sum(axis=1)
    norm = np.zeros_like(cm)
    ok = row_sums > 0
    norm[ok] = cm[ok] / row_sums[ok, None]

    lab_size, txt_size = _find_fontsize(n, annotate)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [_class_label(task, c) for c in classes]
    supports = [int(s) for s in row_sums]
    ax.set_xticklabels(labels, fontsize=lab_size, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(
        [f"{l}\n(n={s})" for l, s in zip(labels, supports)],
        fontsize=lab_size,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotate:
        for i in range(n):
            for j in range(n):
                val = int(cm[i, j])
                if val == 0:
                    continue
                color = "white" if norm[i, j] > 0.55 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=txt_size, color=color)
    ax.set_title(title, fontweight="bold", fontsize=12)
    return im


def _shared_colorbar(fig, ims, axs, label: str = "Recall (fila)"):
    mappable = ims[0]
    cbar = fig.colorbar(mappable, ax=axs, fraction=0.03, pad=0.02)
    cbar.set_label(label, fontsize=10)
    return cbar


def _grid_figure(entries, ncols, cell_in, task, fname, out_dir):
    """entries: list of (title, cm, classes). Grid con ncols columnas."""
    n = len(entries)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_in * ncols + 0.6, cell_in * nrows + 0.5),
                             squeeze=False)
    ims = []
    for k, (title, cm, classes) in enumerate(entries):
        ax = axes[k // ncols][k % ncols]
        im = _plot_cm(ax, cm, classes, task, title)
        ims.append(im)
    _shared_colorbar(fig, ims, list(axes.ravel()))
    fig.tight_layout(h_pad=0.9, w_pad=0.9)
    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fname}")


def _eval_entries(results_dir, task: str, model: str, dataset: str) -> dict:
    """(cm, classes, n_test) del eval json de un (model, task, dataset)."""
    d = _load_json(results_dir / f"eval_{model}_{task}_{dataset}.json")
    p = d["probes"]["mlp"]
    return {
        "cm": p["confusion_matrix"]["confusion_matrix"],
        "classes": list(d["classes"]),
        "n_test": d["n_test"],
    }


def _cross_entries(results_dir, task: str, model: str, train: str, test: str,
                   train_classes: list[str]) -> dict:
    d = _load_json(results_dir / f"cross_{model}_{task}_{train}_to_{test}_mlp.json")
    return {
        "cm": d["confusion_matrix"]["confusion_matrix"],
        "classes": train_classes,
        "n_test": d["n_test"],
        "n_train": d["n_train"],
    }


def _top_confusions(results_dir, model: str, task: str, dataset: str) -> list:
    d = _load_json(results_dir / f"eval_{model}_{task}_{dataset}.json")
    ed = d["probes"]["mlp"].get("error_distribution") or {}
    return ed.get("confused_pairs", [])[:5]


def main():
    parser = argparse.ArgumentParser(description="Matrices de confusión normalizadas del Capítulo 6")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Directorio de salida (default: <Escrito>/images)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _find_results_dir()
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent.parent / "images"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Matrices de confusión del Capítulo 6 ===")
    print(f"  Resultados: {results_dir}")
    print(f"  Salida:     {out_dir}\n")

    # ── Intra-dataset: mama ────────────────────────────────────────────────
    for task, task_label, fname in [
        ("breast_her2", "HER2", "fig6_cm_her2.png"),
        ("breast_molsub", "MolSub", "fig6_cm_molsub.png"),
    ]:
        entries = []
        for model in MODELS:
            for ds in ["BCNB", "HISTAI_Breast", "HistologyHSI"]:
                e = _eval_entries(results_dir, task, model, ds)
                entries.append((f"{MODEL_LABEL[model]}\n{ds.replace('HistologyHSI', 'HSI-BC*')}", e["cm"], e["classes"]))
        _grid_figure(entries, ncols=3, cell_in=1.9, task=task, fname=fname, out_dir=out_dir)

    # ── Intra-dataset: órgano ──────────────────────────────────────────────
    entries = []
    for model in MODELS:
        e = _eval_entries(results_dir, "organ", model, "all")
        entries.append((MODEL_LABEL[model], e["cm"], e["classes"]))
    _grid_figure(entries, ncols=4, cell_in=2.0, task="organ", fname="fig6_cm_organ.png", out_dir=out_dir)

    # ── Intra-dataset: colorrectal (Site 9×9 y Side 5×5) ───────────────────
    for ds, ds_label in [("HISTAI_CRC_B1", "b1"), ("HISTAI_CRC_B2", "b2")]:
        entries = []
        for model in MODELS:
            e = _eval_entries(results_dir, "crc_site", model, ds)
            entries.append((MODEL_LABEL[model], e["cm"], e["classes"]))
        _grid_figure(entries, ncols=2, cell_in=2.5, task="crc_site",
                     fname=f"fig6_cm_site_{ds_label}.png", out_dir=out_dir)

    entries = []
    for model in MODELS:
        for ds, ds_label in [("HISTAI_CRC_B1", "B1"), ("HISTAI_CRC_B2", "B2")]:
            e = _eval_entries(results_dir, "crc_side", model, ds)
            entries.append((f"{MODEL_LABEL[model]}\n{ds_label}", e["cm"], e["classes"]))
    _grid_figure(entries, ncols=2, cell_in=2.1, task="crc_side", fname="fig6_cm_side.png", out_dir=out_dir)

    # ── Cross-dataset: colorrectal ─────────────────────────────────────────
    for train, test, fname in [
        ("HISTAI_CRC_B1", "HISTAI_CRC_B2", "fig6_cm_site_cross_b1_to_b2.png"),
        ("HISTAI_CRC_B2", "HISTAI_CRC_B1", "fig6_cm_site_cross_b2_to_b1.png"),
    ]:
        entries = []
        # Clases del dataset de entrenamiento (eje de la CM cross)
        probe = _eval_entries(results_dir, "crc_site", "uni2", train)
        for model in MODELS:
            e = _cross_entries(results_dir, "crc_site", model, train, test, probe["classes"])
            entries.append((f"{MODEL_LABEL[model]}\nB1->B2" if train == "HISTAI_CRC_B1"
                            else f"{MODEL_LABEL[model]}\nB2->B1", e["cm"], e["classes"]))
        _grid_figure(entries, ncols=2, cell_in=2.5, task="crc_site", fname=fname, out_dir=out_dir)

    entries = []
    probe = _eval_entries(results_dir, "crc_side", "uni2", "HISTAI_CRC_B1")
    for model in MODELS:
        for train, test in [("HISTAI_CRC_B1", "HISTAI_CRC_B2"), ("HISTAI_CRC_B2", "HISTAI_CRC_B1")]:
            e = _cross_entries(results_dir, "crc_side", model, train, test, probe["classes"])
            title = f"{MODEL_LABEL[model]} · B1->B2" if train == "HISTAI_CRC_B1" else f"{MODEL_LABEL[model]} · B2->B1"
            entries.append((title, e["cm"], e["classes"]))
    _grid_figure(entries, ncols=2, cell_in=2.1, task="crc_side", fname="fig6_cm_side_cross.png", out_dir=out_dir)

    # ── Pares más confundidos (para el texto del capítulo) ─────────────────
    print("\n=== Pares más confundidos (texto de análisis) ===")
    for model in MODELS:
        for task, ds in [("crc_site", "HISTAI_CRC_B1"), ("crc_site", "HISTAI_CRC_B2"),
                         ("crc_side", "HISTAI_CRC_B1"), ("crc_side", "HISTAI_CRC_B2")]:
            d = _load_json(results_dir / f"eval_{model}_{task}_{ds}.json")
            classes = list(d["classes"])
            ed = d["probes"]["mlp"].get("error_distribution") or {}
            top = ed.get("confused_pairs", [])[:4]
            if top:
                pairs = ", ".join(
                    f"{_class_label(task, classes[int(t)])}->{_class_label(task, classes[int(p)])} ({c})"
                    for t, p, c in top
                )
                print(f"  {MODEL_LABEL[model]:<14} {task}:{ds:<18} {pairs}")
        for train, test in [("HISTAI_CRC_B1", "HISTAI_CRC_B2"), ("HISTAI_CRC_B2", "HISTAI_CRC_B1")]:
            d = _load_json(results_dir / f"cross_{model}_crc_site_{train}_to_{test}_mlp.json")
            probe = _eval_entries(results_dir, "crc_site", "uni2", train)
            classes = probe["classes"]
            ed = d.get("error_distribution") or {}
            top = ed.get("confused_pairs", [])[:4]
            if top:
                pairs = ", ".join(
                    f"{_class_label('crc_site', classes[int(t)])}->{_class_label('crc_site', classes[int(p)])} ({c})"
                    for t, p, c in top
                )
                print(f"  {MODEL_LABEL[model]:<14} cross-site {train}->{test:<24} {pairs}")

    print(f"\n¡Figuras generadas en {out_dir}!")


if __name__ == "__main__":
    main()
