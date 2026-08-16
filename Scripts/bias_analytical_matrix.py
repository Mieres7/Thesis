#!/usr/bin/env python3
"""
Construye la matriz analítica de bias y genera los cálculos de
representación por subgrupo para los datasets seleccionados.
Productos:
  1. Matriz analítica (CSV): una fila por dataset seleccionado
  2. Tabla de distribución demográfica + variable objetivo
  3. Tablas de contingencia: target x sexo, target x grupo_etario
  4. Figuras de barras apiladas por dominio
"""

import os, sys, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. RUTAS
# ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "datasets_metadata" / "csv"
CLEAN = BASE / "Scripts" / "METADATA_FINAL" / "clean"
OUT = BASE / "Escrito" / "metricas" / "bias_analysis"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# ──────────────────────────────────────────────
# 1. CARGA DE DATOS CURADOS (armonizados)
# ──────────────────────────────────────────────

def load_breast():
    """Carga los datasets de mama con columnas homogéneas."""
    datasets = {}

    # BCNB
    bcnb = pd.read_csv(CLEAN / "breast" / "BCNB.csv")
    bcnb.rename(columns={"id": "case_id"}, inplace=True)
    bcnb["dataset"] = "BCNB"
    bcnb["sex"] = "F"
    bcnb["age"] = pd.to_numeric(bcnb["age"], errors="coerce")
    datasets["BCNB"] = bcnb

    # HISTAI-Breast: base demográfica = cohorte curada completa (n=1489);
    # HER2/Subtipo molecular se anexan por left-join (coberturas parciales: 878/852)
    histai_regroup = pd.read_csv(DATASETS / "histai_breast_metadata_regroup.csv")
    histai_regroup["case_id"] = histai_regroup["case_mapping"].apply(
        lambda x: f"histai/HISTAI-breast/{x}" if not x.startswith("histai/") else x
    )
    sex_map = histai_regroup[["case_id", "gender"]].drop_duplicates("case_id")

    histai_her2 = pd.read_csv(CLEAN / "breast" / "HISTAI_BC_HER2status.csv")
    histai_her2.rename(columns={"case_mapping": "case_id"}, inplace=True)

    # Molecular_subtype via dedicated file (the HER2 file also has it, but less reliable)
    histai_ms = pd.read_csv(CLEAN / "breast" / "HISTAI_BC_MolecularSubtype.csv")
    histai_ms.rename(columns={"case_mapping": "case_id"}, inplace=True)

    histai_breast = histai_regroup[["case_id", "age", "gender"]].copy()
    histai_breast = histai_breast.merge(
        histai_her2[["case_id", "HER2_status"]], on="case_id", how="left"
    )
    histai_breast = histai_breast.merge(
        histai_ms[["case_id", "Molecular_subtype"]], on="case_id", how="left"
    )
    histai_breast["dataset"] = "HISTAI-Breast"
    histai_breast["sex"] = histai_breast["gender"].str.upper()
    histai_breast["age"] = pd.to_numeric(histai_breast["age"], errors="coerce")
    histai_breast["HER2_status"] = pd.to_numeric(histai_breast["HER2_status"], errors="coerce")
    datasets["HISTAI-Breast"] = histai_breast

    # HSI-BRCA (from xlsx Clinical Data sheet)
    xls_path = BASE / "datasets_metadata" / "xlsx" / "7_metadata_HSI-BRCA.xlsx"
    hsi_brca = pd.read_excel(xls_path, sheet_name="Clinical Data")
    hsi_brca.rename(columns={
        "Case ID": "case_id",
        "Age at Diagnosis (Years)": "age",
        "Sex at Birth": "sex",
        "HER2": "HER2_status",
        "Molecular_subtype": "Molecular_subtype",
    }, inplace=True)
    hsi_brca["dataset"] = "HSI-BRCA"
    hsi_brca["age"] = pd.to_numeric(hsi_brca["age"], errors="coerce")
    hsi_brca["sex"] = hsi_brca["sex"].str.upper()
    # Recodificar Molecular_subtype de códigos numéricos a etiquetas
    mol_map = {0: "Luminal A", 1: "Luminal B", 2: "Luminal B",
               3: "HER2+", 4: "Triple Negative"}
    if "Molecular_subtype" in hsi_brca.columns:
        hsi_brca["Molecular_subtype"] = (
            hsi_brca["Molecular_subtype"].map(mol_map)
        )
    datasets["HSI-BRCA"] = hsi_brca

    # HISTAI-Breast subtipo-only (independent of HER2 file)
    histai_ms_indep = pd.read_csv(CLEAN / "breast" / "HISTAI_BC_MolecularSubtype.csv")
    histai_ms_indep.rename(columns={"case_mapping": "case_id"}, inplace=True)
    histai_ms_indep["dataset"] = "HISTAI-Breast"
    histai_ms_indep["HER2_status"] = pd.to_numeric(histai_ms_indep["HER2_status"], errors="coerce")
    histai_ms_indep["age"] = pd.to_numeric(histai_ms_indep["age"], errors="coerce")
    # Merge sex from regroup
    histai_ms_indep = histai_ms_indep.merge(sex_map, on="case_id", how="left")
    histai_ms_indep.rename(columns={"gender": "sex"}, inplace=True)
    histai_ms_indep["sex"] = histai_ms_indep["sex"].str.upper()
    datasets["HISTAI-Breast_Subtipo"] = histai_ms_indep

    return datasets


def load_colorectal():
    """Carga los datasets colorrectales con columnas homogéneas."""
    datasets = {}

    # SurGen combinado (SR386 + SR1482) — site
    sr386 = pd.read_csv(CLEAN / "colorectal" / "site" / "SURGEN368_CRC_site.csv")
    sr386.rename(columns={"is": "case_id"}, inplace=True)
    sr386["dataset"] = "SurGen"
    sr386["sex"] = sr386["sex"].str.upper()
    sr386["age"] = pd.to_numeric(sr386["age"], errors="coerce")

    sr1482 = pd.read_csv(CLEAN / "colorectal" / "site" / "SURGEN1482_CRC_site.csv")
    sr1482.rename(columns={"id": "case_id"}, inplace=True)
    sr1482["dataset"] = "SurGen"
    sr1482["sex"] = sr1482["sex"].str.upper()
    sr1482["age"] = pd.to_numeric(sr1482["age"], errors="coerce")

    surgen = pd.concat([sr386, sr1482], ignore_index=True)

    # SurGen combinado — side
    sr386_side = pd.read_csv(CLEAN / "colorectal" / "side" / "SURGEN368_CRC_side.csv")
    sr386_side.rename(columns={"id": "case_id"}, inplace=True)
    sr386_side["age"] = pd.to_numeric(sr386_side["age"], errors="coerce")
    sr386_side["sex"] = sr386_side["sex"].str.upper()

    sr1482_side = pd.read_csv(CLEAN / "colorectal" / "side" / "SURGEN1482_CRC_side.csv")
    sr1482_side.rename(columns={"id": "case_id"}, inplace=True)
    sr1482_side["age"] = pd.to_numeric(sr1482_side["age"], errors="coerce")
    sr1482_side["sex"] = sr1482_side["sex"].str.upper()

    surgen_side = pd.concat([sr386_side, sr1482_side], ignore_index=True)
    surgen_side["dataset"] = "SurGen"

    datasets["SurGen"] = surgen
    datasets["SurGen_side"] = surgen_side

    # HISTAI-Colorectal-b1 — site
    histai_b1 = pd.read_csv(CLEAN / "colorectal" / "site" / "HISTAI_B1_CRC_site.csv")
    histai_b1.rename(columns={"case_mapping": "case_id"}, inplace=True)
    histai_b1["dataset"] = "HISTAI-CRC-B1"
    histai_b1["sex"] = histai_b1["sex"].str.upper()
    histai_b1["age"] = pd.to_numeric(histai_b1["age"], errors="coerce")
    datasets["HISTAI-CRC-B1"] = histai_b1

    # HISTAI-Colorectal-b1 — side
    histai_b1_side = pd.read_csv(CLEAN / "colorectal" / "side" / "HISTAI_B1_CRC_side.csv")
    histai_b1_side.rename(columns={"case_mapping": "case_id"}, inplace=True)
    histai_b1_side["dataset"] = "HISTAI-CRC-B1"
    histai_b1_side["sex"] = histai_b1_side["sex"].str.upper()
    histai_b1_side["age"] = pd.to_numeric(histai_b1_side["age"], errors="coerce")
    datasets["HISTAI-CRC-B1_side"] = histai_b1_side

    # HISTAI-Colorectal-b2 — site (column is called 'side' but contains site categories)
    histai_b2 = pd.read_csv(CLEAN / "colorectal" / "site" / "HISTAI_B2_CRC_site.csv")
    if "case_mapping" in histai_b2.columns:
        histai_b2.rename(columns={"case_mapping": "case_id"}, inplace=True)
    histai_b2["dataset"] = "HISTAI-CRC-B2"
    if "sex" in histai_b2.columns:
        histai_b2["sex"] = histai_b2["sex"].str.upper()
    if "age" in histai_b2.columns:
        histai_b2["age"] = pd.to_numeric(histai_b2["age"], errors="coerce")
    datasets["HISTAI-CRC-B2"] = histai_b2

    # HISTAI-Colorectal-b2 — side (real side categories)
    histai_b2_side = pd.read_csv(CLEAN / "colorectal" / "side" / "HISTAI_B2_CRC_side.csv")
    histai_b2_side.rename(columns={"case_mapping": "case_id"}, inplace=True)
    histai_b2_side["dataset"] = "HISTAI-CRC-B2"
    histai_b2_side["sex"] = histai_b2_side["sex"].str.upper()
    histai_b2_side["age"] = pd.to_numeric(histai_b2_side["age"], errors="coerce")
    datasets["HISTAI-CRC-B2_side"] = histai_b2_side

    return datasets


# ──────────────────────────────────────────────
# 2. FUNCIONES DE ANÁLISIS
# ──────────────────────────────────────────────

AGE_BINS = [0, 50, 70, 200]
AGE_LABELS = ["<50", "50-70", ">=70"]


def age_group(age):
    if pd.isna(age):
        return "MISSING"
    for i, bound in enumerate(AGE_BINS[:-1]):
        if bound <= age < AGE_BINS[i + 1]:
            return AGE_LABELS[i]
    return AGE_LABELS[-1]


def describe_age(series):
    s = series.dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan,
                "q25": np.nan, "q75": np.nan, "missing_n": 0, "missing_pct": 0}
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    return {
        "n": len(s),
        "mean": round(s.mean(), 1),
        "median": round(s.median(), 1),
        "IQR": f"[{q25:.0f}-{q75:.0f}]",
        "missing_n": int(series.isna().sum()),
        "missing_pct": round(series.isna().mean() * 100, 1),
    }


def sex_distribution(df):
    counts = df["sex"].value_counts(dropna=False)
    total = counts.sum()
    result = {}
    for k, v in counts.items():
        label = k if pd.notna(k) else "MISSING"
        result[f"{label}_n"] = int(v)
        result[f"{label}_pct"] = round(v / total * 100, 1)
    result["total"] = int(total)
    return result


def age_group_distribution(df):
    df = df.copy()
    df["age_grp"] = df["age"].apply(age_group)
    counts = df["age_grp"].value_counts()
    total = counts.sum()
    result = {}
    for g in AGE_LABELS + ["MISSING"]:
        v = counts.get(g, 0)
        result[f"{g}_n"] = int(v)
        result[f"{g}_pct"] = round(v / total * 100, 1) if total else 0
    return result


def target_by_sex(df, target_col):
    """Tabla P(Y | sexo)"""
    df = df.dropna(subset=[target_col])
    if df.empty:
        return None, None
    ct = pd.crosstab(df[target_col], df["sex"], margins=True, margins_name="Total")
    return ct, ct.div(ct.loc["Total"], axis=1) * 100


def target_by_age_group(df, target_col):
    """Tabla P(Y | grupo_etario)"""
    df = df.copy()
    df["age_grp"] = df["age"].apply(age_group)
    df = df.dropna(subset=[target_col])
    if df.empty:
        return None, None
    ct = pd.crosstab(df[target_col], df["age_grp"], margins=True, margins_name="Total")
    return ct, ct.div(ct.loc["Total"], axis=1) * 100


def small_cell_report(ct, threshold=10):
    """Identifica celdas con n < threshold."""
    if ct is None:
        return []
    small = []
    for col in ct.columns:
        for idx in ct.index:
            val = ct.loc[idx, col]
            if val < threshold:
                small.append((idx, col, int(val)))
    return small


# ──────────────────────────────────────────────
# 3. MATRIZ ANALÍTICA
# ──────────────────────────────────────────────

def build_analytical_matrix(breast_dfs, colorectal_dfs):
    """
    Construye la matriz con una fila por dataset seleccionado.
    Columnas: Dominio, Dataset, Cap5-6, Var objetivo, Sexo disp, Edad disp,
              Otras vars clínicas, Escáner, Tinción, Magnificación, Período, Restricción principal
    """
    rows = []

    # -- MAMA --
    # BCNB
    rows.append({
        "Dominio": "Mama",
        "Dataset/cohorte": "BCNB",
        "Se usará en Cap. 5-6": "Sí",
        "Variable objetivo": "HER2 (binario/multiclase), Subtipo molecular (4 clases)",
        "Sexo disponible": "No (100% femenino, inferido)",
        "Edad disponible": "Sí (completo, 0% missing)",
        "Otras variables clínicas": "ER, PR, Ki67, Grado histológico, Tamaño tumoral, Tipo tumoral, Estado ganglionar",
        "Escáner": "No reportado",
        "Tinción": "H&E",
        "Magnificación": "20x",
        "Período": "No reportado",
        "Restricción principal": "Sin sexo para estratificación; solo edad como atributo sensible",
    })

    # HISTAI-Breast
    rows.append({
        "Dominio": "Mama",
        "Dataset/cohorte": "HISTAI-Breast",
        "Se usará en Cap. 5-6": "Sí",
        "Variable objetivo": "HER2 (extraído de texto libre), Subtipo molecular (extraído de texto libre)",
        "Sexo disponible": "Sí (f/m, ~97% femenino)",
        "Edad disponible": "Sí (14.7% missing en metadata completo, pero extraído para subconjunto con HER2)",
        "Otras variables clínicas": "Diagnóstico normalizado, grupo taxonómico, lateralidad (extraídos de texto libre)",
        "Escáner": "Leica Aperio GT450/AT2 (algunos Hamamatsu/3DHISTECH)",
        "Tinción": "H&E (mayoría)",
        "Magnificación": "20x (fracción 40x)",
        "Período": "No reportado",
        "Restricción principal": "HER2/Subtipo extraídos de texto libre -> posible sesgo de extracción; subgrupo masculino muy pequeño",
    })

    # HSI-BRCA
    rows.append({
        "Dominio": "Mama",
        "Dataset/cohorte": "Histology HSI-BRCA-Recurrence",
        "Se usará en Cap. 5-6": "Sí (limitado por tamaño)",
        "Variable objetivo": "HER2, Subtipo molecular (disponibles en metadata CDE)",
        "Sexo disponible": "Sí (Sex at Birth)",
        "Edad disponible": "Sí (Age at Diagnosis)",
        "Otras variables clínicas": "Raza, Etnia, Estado menopáusico, ER, PR, Ki67, Estadio TNM, Tratamiento, Seguimiento, DFS, Supervivencia global",
        "Escáner": "Pannoramic 250 Flash III (3DHISTECH)",
        "Tinción": "H&E (también hiperespectral)",
        "Magnificación": "20x",
        "Período": "2006-2015",
        "Restricción principal": "n=47 -> muestras muy pequeñas para inferencias de fairness; análisis principalmente descriptivo",
    })

    # -- COLORRECTAL --
    # SurGen (SR386 + SR1482 combinados)
    rows.append({
        "Dominio": "Colorrectal",
        "Dataset/cohorte": "SurGen",
        "Se usará en Cap. 5-6": "Sí",
        "Variable objetivo": "Sitio anatómico, Lateralidad",
        "Sexo disponible": "Sí (M/F, balanceado)",
        "Edad disponible": "Sí (0% missing)",
        "Otras variables clínicas": "Estadio (TNM, Dukes), KRAS/NRAS/BRAF, MMR/MSI, Diferenciación, LVI, Supervivencia, Tratamiento pre-op",
        "Escáner": "ZEISS AxioScan.Z1",
        "Tinción": "H&E",
        "Magnificación": "40x",
        "Período": "No reportado",
        "Restricción principal": "Cohorte hospitalaria de un solo centro (NHS Lothian, Escocia); la subcohorte SR1482 incluye casos metastásicos con sitio más fragmentado",
    })

    # HISTAI-CRC-B1
    rows.append({
        "Dominio": "Colorrectal",
        "Dataset/cohorte": "HISTAI-Colorectal-b1",
        "Se usará en Cap. 5-6": "Sí",
        "Variable objetivo": "Sitio anatómico, Lateralidad",
        "Sexo disponible": "Sí (f/m)",
        "Edad disponible": "Sí",
        "Otras variables clínicas": "Diagnóstico normalizado, grupo taxonómico (extraídos de texto libre)",
        "Escáner": "Leica Aperio GT450/AT2 (algunos Hamamatsu/3DHISTECH)",
        "Tinción": "H&E (mayoría)",
        "Magnificación": "20x (fracción 40x)",
        "Período": "No reportado",
        "Restricción principal": "Variables clínicas adicionales solo extraíbles de texto libre; 17.1% missing global",
    })

    # HISTAI-CRC-B2
    rows.append({
        "Dominio": "Colorrectal",
        "Dataset/cohorte": "HISTAI-Colorectal-b2",
        "Se usará en Cap. 5-6": "Sí (limitado por tamaño)",
        "Variable objetivo": "Sitio anatómico, Lateralidad",
        "Sexo disponible": "Sí (f/m)",
        "Edad disponible": "Sí",
        "Otras variables clínicas": "Diagnóstico normalizado (pólipos/adenomas vs carcinoma)",
        "Escáner": "Leica Aperio GT450/AT2 (algunos Hamamatsu/3DHISTECH)",
        "Tinción": "H&E (mayoría)",
        "Magnificación": "20x (fracción 40x)",
        "Período": "No reportado",
        "Restricción principal": "n=57 -> cohorte muy pequeña; predominan lesiones premalignas vs carcinoma invasivo en otras cohortes",
    })

    matrix = pd.DataFrame(rows)
    return matrix


# ──────────────────────────────────────────────
# 4. ANÁLISIS CUANTITATIVO POR DATASET
# ──────────────────────────────────────────────

def analyze_dataset(df, name, target_col, domain):
    """Ejecuta el análisis completo para un dataset."""
    print(f"\n{'='*60}")
    print(f"  {name} (n={len(df)}) — Dominio: {domain}")
    print(f"{'='*60}")

    results = {"dataset": name, "domain": domain, "n_total": len(df)}

    # --- Sexo ---
    if "sex" in df.columns and df["sex"].notna().any():
        sex_stats = sex_distribution(df)
        results["sex"] = sex_stats
        print(f"\n  Sexo: {json.dumps(sex_stats, indent=2)}")
    else:
        print("\n  Sexo: No disponible")
        results["sex"] = None

    # --- Edad ---
    if "age" in df.columns:
        age_stats = describe_age(df["age"])
        results["age"] = age_stats
        print(f"\n  Edad: mean={age_stats['mean']}, median={age_stats['median']}, "
              f"IQR={age_stats['IQR']}, missing={age_stats['missing_pct']}%")

        age_grp = age_group_distribution(df)
        results["age_group"] = age_grp
        print(f"  Grupos etarios: {json.dumps(age_grp, indent=2)}")
    else:
        print("\n  Edad: No disponible")
        results["age"] = None

    # --- Variable objetivo ---
    if target_col and target_col in df.columns:
        y_clean = df[target_col].dropna()
        y_counts = y_clean.value_counts()
        print(f"\n  Variable objetivo ({target_col}): n={len(y_clean)}, "
              f"categorías={list(y_counts.index)}")
        for cat, cnt in y_counts.items():
            print(f"    {cat}: {cnt} ({cnt/len(y_clean)*100:.1f}%)")
        results["target_dist"] = y_counts.to_dict()

        # Contingencia target x sexo
        if "sex" in df.columns and df["sex"].notna().any():
            ct_s, pct_s = target_by_sex(df, target_col)
            results["target_x_sex"] = ct_s.to_dict() if ct_s is not None else None
            if ct_s is not None:
                small = small_cell_report(ct_s)
                if small:
                    print(f"\n  !! Celdas pequeñas (n<10) en targetxsexo:")
                    for idx, col, val in small:
                        if idx != "Total":
                            print(f"    {idx} x {col}: n={val}")

        # Contingencia target x grupo_etario
        ct_a, pct_a = target_by_age_group(df, target_col)
        results["target_x_age"] = ct_a.to_dict() if ct_a is not None else None
        if ct_a is not None:
            small = small_cell_report(ct_a)
            if small:
                print(f"\n  !! Celdas pequeñas (n<10) en targetxgrupo_etario:")
                for idx, col, val in small:
                    if idx != "Total":
                        print(f"    {idx} x {col}: n={val}")

    else:
        print(f"\n  Variable objetivo ({target_col}): No disponible en este archivo")
        results["target_dist"] = None

    return results


# ──────────────────────────────────────────────
# 5. FIGURAS
# ──────────────────────────────────────────────

def plot_sex_distribution(all_results, domain, filename):
    """Barras apiladas de sexo por dataset dentro de un dominio."""
    domain_dfs = [r for r in all_results if r["domain"] == domain and r.get("sex")]
    if not domain_dfs:
        print(f"  [skip] No hay datos de sexo para {domain}")
        return

    fig, axes = plt.subplots(1, len(domain_dfs), figsize=(5 * len(domain_dfs), 4),
                             sharey=True)
    if len(domain_dfs) == 1:
        axes = [axes]

    for ax, r in zip(axes, domain_dfs):
        sex = r["sex"]
        cats = [k.replace("_n", "").replace("_pct", "")
                for k in sex.keys() if k.endswith("_n") and k != "total_n"]
        values = [sex.get(f"{c}_n", 0) for c in cats]
        total = sex.get("total", 1)
        ax.bar(cats, [v / total * 100 for v in values], color=["#4C72B0", "#DD8452", "#aaaaaa"])
        ax.set_title(f"{r['dataset']}\n(n={total})", fontsize=11)
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)

    fig.suptitle(f"Distribución por sexo — Dominio {domain}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {filename}")


def plot_age_distribution(all_results, domain, filename):
    """Barras apiladas de grupos etarios por dataset."""
    domain_dfs = [r for r in all_results if r["domain"] == domain and r.get("age_group")]
    if not domain_dfs:
        print(f"  [skip] No hay datos de edad para {domain}")
        return

    fig, axes = plt.subplots(1, len(domain_dfs), figsize=(5 * len(domain_dfs), 4),
                             sharey=True)
    if len(domain_dfs) == 1:
        axes = [axes]

    colors = {"<50": "#4C72B0", "50-70": "#DD8452", ">=70": "#55A868", "MISSING": "#C44E52"}

    for ax, r in zip(axes, domain_dfs):
        ag = r["age_group"]
        groups = list(colors.keys())
        values = [ag.get(f"{g}_n", 0) for g in groups]
        total = sum(values)
        pcts = [v / total * 100 if total else 0 for v in values]
        bars = ax.bar(groups, pcts, color=[colors[g] for g in groups])
        ax.set_title(f"{r['dataset']}\n(n={total})", fontsize=11)
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)

    fig.suptitle(f"Distribución por grupo etario — Dominio {domain}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {filename}")


# ──────────────────────────────────────────────
# 6. TABLA RESUMEN EN LATEX / CSV
# ──────────────────────────────────────────────

def build_summary_table(all_results):
    """Tabla resumen: Dataset | n | Sexo (F/M) | Edad (media) | Grupos etarios | Objetivo | Celdas pequeñas"""
    rows = []
    for r in all_results:
        sex_str = ""
        if r.get("sex") and "F_n" in r["sex"]:
            f_n = r["sex"].get("F_n", 0)
            m_n = r["sex"].get("M_n", 0)
            f_pct = r["sex"].get("F_pct", 0)
            m_pct = r["sex"].get("M_pct", 0)
            sex_str = f"F: {f_n} ({f_pct}%) / M: {m_n} ({m_pct}%)"

        age_str = ""
        if r.get("age"):
            a = r["age"]
            age_str = f"media={a['mean']}, med={a['median']}, IQR={a['IQR']}, miss={a['missing_pct']}%"

        age_grp_str = ""
        if r.get("age_group"):
            ag = r["age_group"]
            age_grp_str = (f"<50: {ag.get('<50_n',0)} ({ag.get('<50_pct',0)}%) / "
                           f"50-70: {ag.get('50-70_n',0)} ({ag.get('50-70_pct',0)}%) / "
                           f">=70: {ag.get('>=70_n',0)} ({ag.get('>=70_pct',0)}%)")

        rows.append({
            "Dataset": r["dataset"],
            "Dominio": r["domain"],
            "n": r["n_total"],
            "Sexo": sex_str,
            "Edad": age_str,
            "Grupos etarios": age_grp_str,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MATRIZ ANALÍTICA DE BIAS — CAPÍTULO 4")
    print("=" * 60)

    # 1. Cargar datos
    print("\n>> Cargando datos de mama...")
    breast = load_breast()
    for k, v in breast.items():
        print(f"  {k}: {len(v)} registros, columnas={list(v.columns)}")

    print("\n>> Cargando datos colorrectales...")
    colorectal = load_colorectal()
    for k, v in colorectal.items():
        print(f"  {k}: {len(v)} registros, columnas={list(v.columns)}")

    # 2. Matriz analítica
    print("\n>> Construyendo matriz analítica...")
    matrix = build_analytical_matrix(breast, colorectal)
    matrix.to_csv(OUT / "matriz_analitica.csv", index=False)
    print(f"  -> {OUT / 'matriz_analitica.csv'}")
    print(f"  Dimensiones: {matrix.shape[0]} datasets x {matrix.shape[1]} columnas")
    print("\n" + matrix.to_string(index=False))

    # 3. Análisis cuantitativo
    print("\n" + "=" * 60)
    print("  ANÁLISIS CUANTITATIVO POR DATASET")
    print("=" * 60)

    # Para mama: target = HER2_status (en BCNB e HISTAI-Breast)
    all_results = []

    # ── MAMA: variable objetivo primaria = HER2 ──
    # BCNB
    if "HER2_status" in breast["BCNB"].columns:
        r = analyze_dataset(breast["BCNB"], "BCNB", "HER2_status", "Mama")
        all_results.append(r)

    # HISTAI-Breast
    if "HER2_status" in breast["HISTAI-Breast"].columns:
        r = analyze_dataset(breast["HISTAI-Breast"], "HISTAI-Breast", "HER2_status", "Mama")
        all_results.append(r)

    # HSI-BRCA (edad/sexo solamente, HER2 no está en el CSV plano)
    r = analyze_dataset(breast["HSI-BRCA"], "HSI-BRCA", None, "Mama")
    all_results.append(r)

    # ── MAMA: variable objetivo secundaria = Subtipo molecular ──
    if "Molecular_subtype" in breast["BCNB"].columns:
        r = analyze_dataset(breast["BCNB"], "BCNB (Subtipo mol.)", "Molecular_subtype", "Mama")
        all_results.append(r)

    if "Molecular_subtype" in breast["HISTAI-Breast_Subtipo"].columns:
        r = analyze_dataset(breast["HISTAI-Breast_Subtipo"], "HISTAI-Breast (Subtipo mol.)", "Molecular_subtype", "Mama")
        all_results.append(r)

    if "Molecular_subtype" in breast["HSI-BRCA"].columns:
        r = analyze_dataset(breast["HSI-BRCA"], "HSI-BRCA (Subtipo mol.)", "Molecular_subtype", "Mama")
        all_results.append(r)

    # ── COLORRECTAL: variable objetivo primaria = Sitio ──
    # SurGen (combinado)
    if "site" in colorectal["SurGen"].columns:
        r = analyze_dataset(colorectal["SurGen"], "SurGen", "site", "Colorrectal")
        all_results.append(r)

    # HISTAI-CRC-B1
    if "site" in colorectal["HISTAI-CRC-B1"].columns:
        r = analyze_dataset(colorectal["HISTAI-CRC-B1"], "HISTAI-CRC-B1", "site", "Colorrectal")
        all_results.append(r)

    # HISTAI-CRC-B2
    if colorectal["HISTAI-CRC-B2"] is not None and "site" in colorectal["HISTAI-CRC-B2"].columns:
        r = analyze_dataset(colorectal["HISTAI-CRC-B2"], "HISTAI-CRC-B2", "site", "Colorrectal")
        all_results.append(r)
    else:
        r = analyze_dataset(colorectal["HISTAI-CRC-B2"], "HISTAI-CRC-B2", None, "Colorrectal")
        all_results.append(r)

    # ── COLORRECTAL: variable objetivo secundaria = Lateralidad (side) ──
    if "side" in colorectal["SurGen_side"].columns:
        r = analyze_dataset(colorectal["SurGen_side"], "SurGen (Lateralidad)", "side", "Colorrectal")
        all_results.append(r)

    if "side" in colorectal["HISTAI-CRC-B1_side"].columns:
        r = analyze_dataset(colorectal["HISTAI-CRC-B1_side"], "HISTAI-CRC-B1 (Lateralidad)", "side", "Colorrectal")
        all_results.append(r)

    if "side" in colorectal["HISTAI-CRC-B2_side"].columns:
        r = analyze_dataset(colorectal["HISTAI-CRC-B2_side"], "HISTAI-CRC-B2 (Lateralidad)", "side", "Colorrectal")
        all_results.append(r)

    # 4. Guardar resultados individuales
    print("\n>> Guardando resultados...")
    results_path = OUT / "resultados_por_dataset.json"
    serializable = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                sr[k] = {str(kk): vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
            elif isinstance(v, (int, float, str, list)):
                sr[k] = v
            else:
                sr[k] = str(v)[:200]
        serializable.append(sr)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"  -> {results_path}")

    # 5. Tabla resumen (solo resultados primarios, sin duplicados)
    print("\n>> Tabla resumen de distribución demográfica...")
    primary_results = [r for r in all_results if "(" not in r["dataset"]]
    summary = build_summary_table(primary_results)
    summary.to_csv(OUT / "tabla_distribucion_demografica.csv", index=False)
    print(f"  -> {OUT / 'tabla_distribucion_demografica.csv'}")
    print("\n" + summary.to_string(index=False))

    # 6. Tabla de trazabilidad de muestras
    print("\n>> Tabla de trazabilidad de muestras...")
    trace_rows = []
    trace_data = [
        {"dataset": "BCNB",          "raw": 1058, "curated": 1058, "task": "HER2",        "n_elegible": 1058},
        {"dataset": "BCNB",          "raw": 1058, "curated": 1058, "task": "Subtipo mol.", "n_elegible": 1058},
        {"dataset": "HISTAI-Breast", "raw": 1489, "curated": 1489, "task": "HER2",        "n_elegible": 878},
        {"dataset": "HISTAI-Breast", "raw": 1489, "curated": 1489, "task": "Subtipo mol.", "n_elegible": 852},
        {"dataset": "HSI-BRCA",      "raw": 47,   "curated": 47,   "task": "HER2",        "n_elegible": 47},
        {"dataset": "HSI-BRCA",      "raw": 47,   "curated": 47,   "task": "Subtipo mol.", "n_elegible": 47},
        {"dataset": "SurGen",        "raw": 843,  "curated": 841,  "task": "Sitio",       "n_elegible": 841},
        {"dataset": "SurGen",        "raw": 843,  "curated": 841,  "task": "Lateralidad", "n_elegible": 707},
        {"dataset": "HISTAI-CRC-B1", "raw": 878,  "curated": 878,  "task": "Sitio",       "n_elegible": 878},
        {"dataset": "HISTAI-CRC-B1", "raw": 878,  "curated": 878,  "task": "Lateralidad", "n_elegible": 615},
        {"dataset": "HISTAI-CRC-B2", "raw": 57,   "curated": 57,   "task": "Sitio",       "n_elegible": 57},
        {"dataset": "HISTAI-CRC-B2", "raw": 57,   "curated": 57,   "task": "Lateralidad", "n_elegible": 46},
    ]
    for info in trace_data:
        ret_cur = round(info["n_elegible"] / info["curated"] * 100, 1)
        ret_raw = round(info["n_elegible"] / info["raw"] * 100, 1)
        trace_rows.append({
            "Dataset": info["dataset"],
            "Tarea": info["task"],
            "N_raw": info["raw"],
            "N_cohorte_curada": info["curated"],
            "N_elegible_por_tarea": info["n_elegible"],
            "%_retenido_raw": ret_raw,
            "%_retenido_curated": ret_cur,
        })
    trace_df = pd.DataFrame(trace_rows)
    trace_df.to_csv(OUT / "trazabilidad_muestras.csv", index=False)
    print(f"  -> {OUT / 'trazabilidad_muestras.csv'}")
    print("\n" + trace_df.to_string(index=False))

    # 6b. Comparación demográfica pre/post filtro por tarea (HISTAI-Breast)
    print("\n>> Evaluación de selección inducida por filtrado (HISTAI-Breast)...")
    filter_impact_rows = []
    # Cargar metadata curada completa para comparación
    histai_all = pd.read_csv(BASE / "datasets_metadata" / "csv" / "histai_breast_metadata_regroup.csv")
    histai_all.rename(columns={"case_mapping": "case_id"}, inplace=True)
    # Edad en data curada completa
    if "age" in histai_all.columns:
        age_all = histai_all["age"].dropna()
        desc_all = {
            "dataset": "HISTAI-Breast",
            "filtro": "Pre-filtro (cohorte curada)",
            "n": len(histai_all),
            "n_con_edad": len(age_all),
            "media_edad": round(age_all.mean(), 1),
            "mediana_edad": round(age_all.median(), 1),
            "missing_edad_pct": round((1 - len(age_all) / len(histai_all)) * 100, 1),
            "<50_n": int((age_all < 50).sum()),
            "50-70_n": int(((age_all >= 50) & (age_all < 70)).sum()),
            ">=70_n": int((age_all >= 70).sum()),
        }
        filter_impact_rows.append(desc_all)
    # Post-filtro HER2 (cohorte elegible por HER2, n=878; independiente del merge anterior)
    her2 = pd.read_csv(CLEAN / "breast" / "HISTAI_BC_HER2status.csv")
    her2.rename(columns={"case_mapping": "case_id"}, inplace=True)
    if "age" in her2.columns:
        age_h = her2["age"].dropna()
        desc_h = {
            "dataset": "HISTAI-Breast",
            "filtro": "Post-filtro HER2 (N=878)",
            "n": len(her2),
            "n_con_edad": len(age_h),
            "media_edad": round(age_h.mean(), 1),
            "mediana_edad": round(age_h.median(), 1),
            "missing_edad_pct": round((1 - len(age_h) / len(her2)) * 100, 1),
            "<50_n": int((age_h < 50).sum()),
            "50-70_n": int(((age_h >= 50) & (age_h < 70)).sum()),
            ">=70_n": int((age_h >= 70).sum()),
        }
        filter_impact_rows.append(desc_h)
    # Post-filtro subtipo
    sub = breast["HISTAI-Breast_Subtipo"]
    if "age" in sub.columns:
        age_s = sub["age"].dropna()
        desc_s = {
            "dataset": "HISTAI-Breast",
            "filtro": "Post-filtro Subtipo (N=852)",
            "n": len(sub),
            "n_con_edad": len(age_s),
            "media_edad": round(age_s.mean(), 1),
            "mediana_edad": round(age_s.median(), 1),
            "missing_edad_pct": round((1 - len(age_s) / len(sub)) * 100, 1),
            "<50_n": int((age_s < 50).sum()),
            "50-70_n": int(((age_s >= 50) & (age_s < 70)).sum()),
            ">=70_n": int((age_s >= 70).sum()),
        }
        filter_impact_rows.append(desc_s)

    if filter_impact_rows:
        filter_df = pd.DataFrame(filter_impact_rows)
        filter_df.to_csv(OUT / "impacto_filtro_histai_edad.csv", index=False)
        print(f"  -> {OUT / 'impacto_filtro_histai_edad.csv'}")
        print("\n" + filter_df.to_string(index=False))
    print()

    # 7. Figuras (solo resultados primarios)
    print("\n>> Generando figuras...")
    plot_sex_distribution(primary_results, "Mama", "fig_sexo_mama.png")
    plot_sex_distribution(primary_results, "Colorrectal", "fig_sexo_colorrectal.png")
    plot_age_distribution(primary_results, "Mama", "fig_edad_mama.png")
    plot_age_distribution(primary_results, "Colorrectal", "fig_edad_colorrectal.png")

    # 7. Tablas de contingencia en detalle (para apéndice)
    print("\n>> Tablas de contingencia detalladas...")
    contingency_path = OUT / "contingencias"
    contingency_path.mkdir(exist_ok=True)

    for r in all_results:
        name = r["dataset"]

        # target x sexo
        if r.get("target_x_sex"):
            ct_df = pd.DataFrame(r["target_x_sex"])
            ct_df.to_csv(contingency_path / f"{name}_target_x_sexo.csv")
            print(f"  -> {name}_target_x_sexo.csv")

        # target x edad
        if r.get("target_x_age"):
            ct_df = pd.DataFrame(r["target_x_age"])
            ct_df.to_csv(contingency_path / f"{name}_target_x_edad.csv")
            print(f"  -> {name}_target_x_edad.csv")

    print(f"\n{'='*60}")
    print("  ANÁLISIS COMPLETO")
    print(f"  Productos generados en: {OUT}")
    print(f"{'='*60}")
