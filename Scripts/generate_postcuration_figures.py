#!/usr/bin/env python3
"""
Genera figuras post-curación a partir de los datos en METADATA_FINAL/clean/.
Estilo visual coherente con DatasetAnalyzer.

Figuras generadas:
  1. SurGen unificado: sitio tumoral (site_group_norm)
  2. SurGen unificado: lateralidad (side_norm)
  3. BCNB: subtipo molecular (opcional, regenerado desde datos curados)
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
CLEAN = BASE / "Scripts" / "METADATA_FINAL" / "clean"
OUTPUT = BASE / "Escrito" / "images"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Estilo editorial (coincide con DatasetAnalyzer) ──────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
})

FIXED_PALETTE = ["#4C72B0", "#DD8452", "#66c2a5", "#fc8d62", "#8da0cb", "#b0b0b0"]
CAT_ORDER_SITE = [
    "rectum", "sigmoid", "right_colon", "left_colon",
    "transverse_colon", "rectosigmoid", "colon_unspecified",
    "appendix", "metastatic_liver", "metastatic_lung",
    "metastatic_peritoneal", "other"
]
CAT_ORDER_SIDE = [
    "left", "right", "rectum", "transverse", "rectosigmoid"
]


def load_surgen_site():
    sr386 = pd.read_csv(CLEAN / "colorectal" / "site" / "SURGEN368_CRC_site.csv")
    sr1482 = pd.read_csv(CLEAN / "colorectal" / "site" / "SURGEN1482_CRC_site.csv")
    site_col = "site"
    sr386 = sr386.rename(columns={"is": "id"}) if "is" in sr386.columns else sr386
    sr386["dataset"] = "SR386"
    sr1482["dataset"] = "SR1482"
    combined = pd.concat([
        sr386[["dataset", site_col]].rename(columns={site_col: "site"}),
        sr1482[["dataset", site_col]].rename(columns={site_col: "site"})
    ], ignore_index=True)
    return combined


def load_surgen_side():
    sr386 = pd.read_csv(CLEAN / "colorectal" / "side" / "SURGEN368_CRC_side.csv")
    sr1482 = pd.read_csv(CLEAN / "colorectal" / "side" / "SURGEN1482_CRC_side.csv")
    sr386["dataset"] = "SR386"
    sr1482["dataset"] = "SR1482"
    combined = pd.concat([
        sr386[["dataset", "side"]],
        sr1482[["dataset", "side"]]
    ], ignore_index=True)
    return combined


def plot_grouped_bars(df, cat_col, group_col, cat_order, output_name, xlabel, title):
    n_groups = len(df[group_col].unique())
    colors = FIXED_PALETTE[:n_groups]

    # Build count table
    counts = df.groupby([cat_col, group_col]).size().unstack(fill_value=0)
    counts = counts.reindex([c for c in cat_order if c in counts.index])
    counts = counts.fillna(0).astype(int)
    cats = list(counts.index)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cats))
    w = 0.3
    offsets = np.linspace(-w, w, n_groups)

    for i, grp in enumerate(counts.columns):
        vals = counts[grp].values
        bars = ax.bar(x + offsets[i], vals, w, label=grp, color=colors[i],
                      edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals)*0.01,
                        str(v), ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in cats], rotation=30, ha="right")
    ax.set_ylabel("Frecuencia")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, title="Cohorte")
    ax.margins(y=0.15)

    total = len(df)
    ax.annotate(f"n = {total}", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    plt.tight_layout()
    plt.savefig(OUTPUT / output_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_name}")


def plot_single_bar(df, col, output_name, xlabel, title, cat_order=None):
    counts = df[col].value_counts()
    if cat_order:
        counts = counts.reindex([c for c in cat_order if c in counts.index]).dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = FIXED_PALETTE[:len(counts)]
    bars = ax.bar(range(len(counts)), counts.values, color=colors,
                  edgecolor="white", linewidth=0.6)

    total = counts.sum()
    for bar, v in zip(bars, counts.values):
        pct = 100 * v / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{v} ({pct:.0f}%)", ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xticks(range(len(counts)))
    labels = [str(c).replace("_", " ") for c in counts.index]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Frecuencia")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.margins(y=0.18)

    ax.annotate(f"n = {total}", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    plt.tight_layout()
    plt.savefig(OUTPUT / output_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_name}")


def generate_breast_figures():
    # BCNB molecular subtype
    bcnb = pd.read_csv(CLEAN / "breast" / "BCNB.csv")
    plot_single_bar(
        bcnb, "Molecular_subtype",
        "fig45-bcnb-molecular-subtype-curated.png",
        "Subtipo Molecular",
        "Distribución de subtipo molecular en BCNB (post-curación)"
    )

    # BCNB age
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(bcnb["age"].dropna(), bins=25, color=FIXED_PALETTE[0],
            edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.axvline(bcnb["age"].mean(), color=FIXED_PALETTE[1], linestyle="--",
               linewidth=1.4, label=f"Media = {bcnb['age'].mean():.1f}")
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de edad en BCNB (post-curación)", fontweight="bold")
    ax.legend(frameon=False)
    ax.annotate(f"n = {len(bcnb)}", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig46-bcnb-age-curated.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: fig46-bcnb-age-curated.png")


def main():
    print("Generando figuras post-curación...")
    print(f"Output: {OUTPUT}\n")

    # ── 1. SurGen sitio tumoral unificado ──
    print("[1/4] SurGen sitio tumoral unificado")
    surgen_site = load_surgen_site()
    plot_grouped_bars(
        surgen_site, "site", "dataset", CAT_ORDER_SITE,
        "fig45-surgen-site-curated.png",
        "Sitio tumoral",
        "Distribución del sitio tumoral en SurGen (post-curación)"
    )

    # ── 2. SurGen lateralidad unificada ──
    print("[2/4] SurGen lateralidad unificada")
    surgen_side = load_surgen_side()
    plot_grouped_bars(
        surgen_side, "side", "dataset", CAT_ORDER_SIDE,
        "fig46-surgen-side-curated.png",
        "Lateralidad (side_norm)",
        "Distribución de lateralidad en SurGen (post-curación)"
    )

    # ── 3. BCNB subtipo molecular ──
    print("[3/4] BCNB subtipo molecular")
    generate_breast_figures()

    # ── 4. SurGen sitio total (un solo barra, ambas cohortes) ──
    print("[4/5] SurGen sitio tumoral total")
    surgen_all_site = surgen_site.copy()
    plot_single_bar(
        surgen_all_site, "site",
        "fig47-surgen-site-total-curated.png",
        "Sitio tumoral",
        "Distribución del sitio tumoral en SurGen (ambas cohortes)"
    )

    # ── 5. SurGen lateralidad total (un solo barra, ambas cohortes) ──
    print("[5/5] SurGen lateralidad total")
    surgen_all_side = surgen_side.copy()
    plot_single_bar(
        surgen_all_side, "side",
        "fig48-surgen-side-total-curated.png",
        "Lateralidad",
        "Distribución de lateralidad en SurGen (ambas cohortes)"
    )

    print(f"\n¡Listo! {len(list(OUTPUT.glob('fig4*-curated.png')))} figuras generadas en {OUTPUT}")


if __name__ == "__main__":
    main()
