# Regenera las 3 datasets que contienen las 4 figuras del escrito y las copia.
import os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("IMAGES_DRIVE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "imagesDrive")))

import matplotlib
matplotlib.use("Agg")

from AnalysisConfig import AnalysisConfig
from DatasetAnalyzer import DatasetAnalyzer

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_CSV = os.path.join(ROOT, "datasets_metadata", "csv")

configs = [
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "2_metadata_BCNB.csv"),
        output_dir="outputs", dataset_name="BCNB",
        exclude_columns=["Patient ID"],
        rename_columns={"Age(years)": "Age"},
        custom_missing_by_column={"recurrent date": ["no recurrence"], "Date of death": ["survival"]},
        force_categorical=["Histological grading", "Number of lymph node metastases"],
        column_title_map={"Age": "Edad (años)", "Molecular subtype": "Subtipo Molecular"},
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "histai_breast_metadata_regroup.csv"),
        output_dir="outputs", dataset_name="HISTAI_BREAST",
        exclude_columns=["conclusion", "case_mapping", "icd10"],
        force_numeric=["age"],
        force_categorical=["diagnosis", "diff_diagnostic", "grossing", "micro_protocol", "gender", "specialization"],
        column_title_map={"age": "Edad (años)"},
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "6.1_metadata_SR386_labels.csv"),
        output_dir="outputs", dataset_name="SR368",
        exclude_columns=["case_id"], id_columns=["case_id"],
        force_categorical=["died_within_5_years", "mmr_loss_binary", "stage", "site_of_tumour"],
        column_title_map={"site_of_tumour": "Sitio del Tumor Primario"},
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
]

for cfg in configs:
    print(f"Analizando: {cfg.dataset_name}")
    try:
        analyzer = DatasetAnalyzer(cfg)
        reports = analyzer.run()
        print(f"  OK - {cfg.dataset_name}")
    except Exception as e:
        print(f"  ERROR - {cfg.dataset_name}: {e}")

# Copiar las 4 figuras a Escrito/images/
IMAGES_DIR = os.path.join(ROOT, "Escrito", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

DRIVE = os.environ["IMAGES_DRIVE"]
copies = [
    (os.path.join(DRIVE, "BCNB", "plots_numeric", "Age.png"), os.path.join(IMAGES_DIR, "fig41-bcnb-age.png")),
    (os.path.join(DRIVE, "HISTAI_BREAST", "plots_numeric", "age.png"), os.path.join(IMAGES_DIR, "fig42-histai-breast-age.png")),
    (os.path.join(DRIVE, "BCNB", "plots_categorical", "Molecular subtype.png"), os.path.join(IMAGES_DIR, "fig43-bcnb-molecular-subtype.png")),
    (os.path.join(DRIVE, "SR368", "plots_categorical", "site_of_tumour.png"), os.path.join(IMAGES_DIR, "fig44-sr368-site-of-tumour.png")),
]
for src, dst in copies:
    shutil.copy2(src, dst)
    print(f"Copiado: {dst}")

print("Listo.")
