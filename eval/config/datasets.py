"""
eval/config/datasets.py

Centralizado de DatasetSpecs para todos los datasets y tareas.
Para agregar un dataset o tarea, solo agregá acá.
"""
from __future__ import annotations

from pathlib import Path
from eval.config.schema import DatasetSpec

# ── Ruta base de embeddings (ajustar según el cluster) ────────────────
_EMBED_BASE = "/home/DIINF/datasets_iacis"
_META_BASE = Path(__file__).resolve().parents[2] / "metadata_clean"


# ═══════════════════════════════════════════════════════════════════════
#  BREAST — HER2 Status
# ═══════════════════════════════════════════════════════════════════════

BCNB_HER2 = DatasetSpec(
    name="BCNB",
    metadata_csv=str(_META_BASE / "breast" / "BCNB.csv"),
    id_column="id",
    subfolder_structure="flat",
    embeddings_dir=f"{_EMBED_BASE}/2026_BCNB/embeddings/BCNB",
    label_column="HER2_status",
    sex_column="sex",
    age_column="age",
)

HISTAI_BREAST_HER2 = DatasetSpec(
    name="HISTAI_Breast",
    metadata_csv=str(_META_BASE / "breast" / "HISTAI_BC_HER2status.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"/home/shared_data/HISTAI/embeddings/HISTAI_Breast_embeddins/HISTAI_Breast",
    label_column="HER2_status",
    sex_column="sex",
    age_column="age",
)

HISTOLOGY_HER2 = DatasetSpec(
    name="HistologyHSI",
    metadata_csv=str(_META_BASE / "breast" / "Histology.csv"),
    id_column="id",
    subfolder_structure="flat",
    embeddings_dir=f"{_EMBED_BASE}/2026_HistologyHSI-BC-Recurrence/embeddings/HistologyHSI",
    label_column="HER2_status",
    sex_column="sex",
    age_column="age",
)


# ═══════════════════════════════════════════════════════════════════════
#  BREAST — Molecular Subtype
# ═══════════════════════════════════════════════════════════════════════

BCNB_MOLSUB = DatasetSpec(
    name="BCNB",
    metadata_csv=str(_META_BASE / "breast" / "BCNB.csv"),
    id_column="id",
    subfolder_structure="flat",
    embeddings_dir=f"{_EMBED_BASE}/2026_BCNB/embeddings/BCNB",
    label_column="Molecular_subtype",
    label_map={"HER2(+)": "HER2+", "Triple negative": "Triple Negative"},
    sex_column="sex",
    age_column="age",
)

HISTAI_BREAST_MOLSUB = DatasetSpec(
    name="HISTAI_Breast",
    metadata_csv=str(_META_BASE / "breast" / "HISTAI_BC_MolecularSubtype.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"/home/shared_data/HISTAI/embeddings/HISTAI_Breast_embeddins/HISTAI_Breast",
    label_column="Molecular_subtype",
    label_map={"Triple negative": "Triple Negative"},
    sex_column="sex",
    age_column="age",
)

HISTOLOGY_MOLSUB = DatasetSpec(
    name="HistologyHSI",
    metadata_csv=str(_META_BASE / "breast" / "Histology.csv"),
    id_column="id",
    subfolder_structure="flat",
    embeddings_dir=f"{_EMBED_BASE}/2026_HistologyHSI-BC-Recurrence/embeddings/HistologyHSI",
    label_column="Molecular_subtype",
    label_map={"HER2(+)": "HER2+"},
    sex_column="sex",
    age_column="age",
)


# ═══════════════════════════════════════════════════════════════════════
#  COLORECTAL — Site
# ═══════════════════════════════════════════════════════════════════════

HISTAI_CRC_B1_SITE = DatasetSpec(
    name="HISTAI_CRC_B1",
    metadata_csv=str(_META_BASE / "colorectal" / "site" / "HISTAI_B1_CRC_site.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"/home/shared_data/HISTAI/embeddings/HISTAI_Colorectal_b1_embeddins/HISTAI_colorectal_b1",
    label_column="site",
    sex_column="sex",
    age_column="age",
)

HISTAI_CRC_B2_SITE = DatasetSpec(
    name="HISTAI_CRC_B2",
    metadata_csv=str(_META_BASE / "colorectal" / "site" / "HISTAI_B2_CRC_site.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"{_EMBED_BASE}/2026_HISTAI/embeddings/HISTAI_colorectal_b2_embeddings/HISTAI_colorectal_b2",
    label_column="side",  # NOTE: columna se llama "side" en el CSV pero es site
    sex_column="sex",
    age_column="age",
)

SURGEN_SITE = DatasetSpec(
    name="SurGen",
    metadata_csv=str(_META_BASE / "colorectal" / "site" / "SURGEN_CRC_site.csv"),
    id_column="id",
    subfolder_structure="surgen",
    embeddings_dir=f"/mnt/data1/vmieres/SurGen/embeddings/SurGen",
    label_column="label",
    # Harmonización: clases raras (n<=12, solo en SurGen SR1482) colapsadas
        # a "other" para alinear el vocabulario site con HISTAI_CRC_B1 (9 clases).
        label_map={
            "metastatic_peritoneal": "other",
            "appendix": "other",
            "metastatic_lung": "other",
        },
    sex_column="sex",
    age_column="age",
)


# ═══════════════════════════════════════════════════════════════════════
#  COLORECTAL — Side
# ═══════════════════════════════════════════════════════════════════════

HISTAI_CRC_B1_SIDE = DatasetSpec(
    name="HISTAI_CRC_B1",
    metadata_csv=str(_META_BASE / "colorectal" / "side" / "HISTAI_B1_CRC_side.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"/home/shared_data/HISTAI/embeddings/HISTAI_Colorectal_b1_embeddins/HISTAI_colorectal_b1",
    label_column="side",
    sex_column="sex",
    age_column="age",
)

HISTAI_CRC_B2_SIDE = DatasetSpec(
    name="HISTAI_CRC_B2",
    metadata_csv=str(_META_BASE / "colorectal" / "side" / "HISTAI_B2_CRC_side.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"{_EMBED_BASE}/2026_HISTAI/embeddings/HISTAI_colorectal_b2_embeddings/HISTAI_colorectal_b2",
    label_column="side",
    sex_column="sex",
    age_column="age",
)

SURGEN_SIDE = DatasetSpec(
    name="SurGen",
    metadata_csv=str(_META_BASE / "colorectal" / "side" / "SURGEN_CRC_side.csv"),
    id_column="id",
    subfolder_structure="surgen",
    embeddings_dir="/mnt/data1/vmieres/SurGen/embeddings/SurGen",
    label_column="label",
    sex_column="sex",
    age_column="age",
)


# ═══════════════════════════════════════════════════════════════════════
#  ORGAN — Breast vs Colorectal (tarea transversal)
# ═══════════════════════════════════════════════════════════════════════
# Cada DatasetSpec apunta a los mismos embeddings que las tareas
# originales, pero force_label reemplaza la columna del CSV.

ORGAN_BCNB = DatasetSpec(
    name="BCNB",
    metadata_csv=str(_META_BASE / "breast" / "BCNB.csv"),
    id_column="id",
    subfolder_structure="flat",
    embeddings_dir=f"{_EMBED_BASE}/2026_BCNB/embeddings/BCNB",
    label_column=None,
    force_label="breast",
    sex_column="sex",
    age_column="age",
)

ORGAN_HISTAI_BREAST = DatasetSpec(
    name="HISTAI_Breast",
    metadata_csv=str(_META_BASE / "breast" / "HISTAI_BC_HER2status.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"/home/shared_data/HISTAI/embeddings/HISTAI_Breast_embeddins/HISTAI_Breast",
    label_column=None,
    force_label="breast",
    sex_column="sex",
    age_column="age",
)

ORGAN_HSIBC = DatasetSpec(
    name="HistologyHSI",
    metadata_csv=str(_META_BASE / "breast" / "Histology.csv"),
    id_column="id",
    subfolder_structure="flat",
    embeddings_dir=f"{_EMBED_BASE}/2026_HistologyHSI-BC-Recurrence/embeddings/HistologyHSI",
    label_column=None,
    force_label="breast",
    sex_column="sex",
    age_column="age",
)

ORGAN_HISTAI_CRC_B1 = DatasetSpec(
    name="HISTAI_CRC_B1",
    metadata_csv=str(_META_BASE / "colorectal" / "site" / "HISTAI_B1_CRC_site.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"/home/shared_data/HISTAI/embeddings/HISTAI_Colorectal_b1_embeddins/HISTAI_colorectal_b1",
    label_column=None,
    force_label="colorectal",
    sex_column="sex",
    age_column="age",
)

ORGAN_HISTAI_CRC_B2 = DatasetSpec(
    name="HISTAI_CRC_B2",
    metadata_csv=str(_META_BASE / "colorectal" / "site" / "HISTAI_B2_CRC_site.csv"),
    id_column="case_mapping",
    subfolder_structure="case_folders",
    embeddings_dir=f"{_EMBED_BASE}/2026_HISTAI/embeddings/HISTAI_colorectal_b2_embeddings/HISTAI_colorectal_b2",
    label_column=None,
    force_label="colorectal",
    sex_column="sex",
    age_column="age",
)

ORGAN_SURGEN = DatasetSpec(
    name="SurGen",
    metadata_csv=str(_META_BASE / "colorectal" / "site" / "SURGEN_CRC_site.csv"),
    id_column="id",
    subfolder_structure="surgen",
    embeddings_dir=f"/mnt/data1/vmieres/SurGen/embeddings/SurGen",
    label_column=None,
    force_label="colorectal",
    sex_column="sex",
    age_column="age",
)


# ═══════════════════════════════════════════════════════════════════════
#  REGISTRO DE TAREAS (para scripts)
# ═══════════════════════════════════════════════════════════════════════

TASKS = {
    # ── Breast ──
    "breast_her2": {
        "task_name": "her2_status",
        "datasets": [BCNB_HER2, HISTAI_BREAST_HER2, HISTOLOGY_HER2],
    },
    "breast_molsub": {
        "task_name": "molecular_subtype",
        "datasets": [BCNB_MOLSUB, HISTAI_BREAST_MOLSUB, HISTOLOGY_MOLSUB],
    },
    # ── Colorectal ──
    "crc_site": {
        "task_name": "site",
        "datasets": [HISTAI_CRC_B1_SITE, HISTAI_CRC_B2_SITE, SURGEN_SITE],
    },
    "crc_side": {
        "task_name": "side",
        "datasets": [HISTAI_CRC_B1_SIDE, HISTAI_CRC_B2_SIDE, SURGEN_SIDE],
    },
    # ── Organ ──
    "organ": {
        "task_name": "organ",
        "evaluate_merged": True,  # evaluar todo el pkl como una población única
        "datasets": [
            ORGAN_BCNB, ORGAN_HISTAI_BREAST, ORGAN_HSIBC,
            ORGAN_HISTAI_CRC_B1, ORGAN_HISTAI_CRC_B2,
            ORGAN_SURGEN,
        ],
    },
}

MODELS = ["uni2", "virchow2", "phikon_v2", "prov_gigapath"]


# ═══════════════════════════════════════════════════════════════════════
#  DECISIÓN EXPLÍCITA CV vs. SPLIT FIJO (por dataset)
# ═══════════════════════════════════════════════════════════════════════
# La profesora confirmó que el análisis de fairness por subgrupos
# (sexo/edad) requiere evaluar sobre el 100% de los pacientes de cada
# cohorte para que las celdas subgrupo × clase tengan soporte suficiente
# (umbral n >= 10). Eso exige validación cruzada 5-fold (OOF combinadas)
# en TODAS las cohortes que alimentan ese análisis.
#
# Esta tabla es LA ÚNICA fuente de la decisión. La función de decisión
# vive en eval/evaluation/splitter.py (should_use_cv / choose_eval_method)
# y NINGÚN otro módulo debe hardcodear "if n < threshold".
#
#   - "use_cv": True  -> 5-fold CV (StratifiedGroupKFold, agrupo por uid)
#   - "use_cv": False -> split fijo 70/15/15 (split_train_val_test)
#   - "reason":        human-readable motivo (fairness_subgroups, small_n, ...)
#   - "method_id":     opcional, explícito en resultados (cv5 | fixed_split)
#
# Datasets NO presentes acá usan el fallback por tamaño
# (SMALL_DATASET_THRESHOLD en splitter.py).
DATASET_EVAL_CONFIG = {
    # ── Cambian de split fijo → CV (fairness por subgrupo sobre el 100%) ──
    "BCNB":            {"use_cv": True, "reason": "fairness_subgroups"},
    "HISTAI_Breast":   {"use_cv": True, "reason": "fairness_subgroups"},
    "HISTAI_CRC_B1":   {"use_cv": True, "reason": "fairness_subgroups"},
    # ── SurGen fusionado (SR1482 + SR386 curado/armonizado como una cohorte) ──
    "SurGen":          {"use_cv": True, "reason": "fairness_subgroups"},
    # ── Ya usaban CV 5-fold antes de este cambio (no cambian) ─────────────
    "HISTAI_CRC_B2":   {"use_cv": True, "reason": "small_n"},
    "HistologyHSI":    {"use_cv": True, "reason": "small_n"},
}
