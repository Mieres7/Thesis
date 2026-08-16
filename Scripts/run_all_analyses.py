# Script para ejecutar todos los análisis de datasets y generar imágenes en imagesDrive/
# Uso: $env:IMAGES_DRIVE="C:\Users\vicen\Repositories\Tesis_general\imagesDrive"; python Scripts/run_all_analyses.py

import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("IMAGES_DRIVE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "imagesDrive")))

import matplotlib
matplotlib.use("Agg")

from AnalysisConfig import AnalysisConfig
from DatasetAnalyzer import DatasetAnalyzer

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_CSV = os.path.join(ROOT, "datasets_metadata", "csv")
DATA_XLSX = os.path.join(ROOT, "datasets_metadata", "xlsx")

configs = [
    # 1. Ovarian
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "1_metadata_ovarian_1.csv"),
        output_dir="outputs", dataset_name="1_ovarian_metadata",
        exclude_columns=["Patient ID"], id_columns=["No."],
        force_numeric=["Age", "BMI", "number of avastin administration"],
        force_categorical=["Diagnosis", "FIGO stage", "operation", "method for avastin use"],
        custom_missing_by_column={"Date of death": ["alive"], "Date of recurrence": ["no recurrence", "nt"]},
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # 2. BCNB
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
    # 3. CLWD
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "3_metadata_CLWD.csv"),
        output_dir="outputs", dataset_name="CLWD",
        exclude_columns=["SampleNumber", "WSI_ID"], id_columns=["SampleNumber"],
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # 4. NDB-UFES
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "4_metadata_ndb-ufes.csv"),
        output_dir="outputs", dataset_name="NDB_UFES",
        exclude_columns=["lesion_id", "patient_id"], id_columns=["public_id"],
        force_categorical=["age_group"],
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # 5. TDBTA
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "5_metadata_TDBTA.csv"),
        output_dir="outputs", dataset_name="TDBTA",
        exclude_columns=["uuid", "pat_id"], id_columns=["public_id"],
        force_categorical=["diagnosis", "location", "control", "recurrence"],
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # 6.1 SurGen-SR386
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "6.1_metadata_SR386_labels.csv"),
        output_dir="outputs", dataset_name="SR368",
        exclude_columns=["case_id"], id_columns=["case_id"],
        force_categorical=["died_within_5_years", "mmr_loss_binary", "stage", "site_of_tumour"],
        column_title_map={"site_of_tumour": "Sitio del Tumor Primario"},
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # 6.2 SurGen-SR1482
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "6.2_metadata_SR1482_labels.csv"),
        output_dir="outputs", dataset_name="SR1482",
        exclude_columns=["case_id"], id_columns=["case_id"],
        force_categorical=["tumour_site"],
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # 7. HSI-BC
    AnalysisConfig(
        file_path=os.path.join(DATA_XLSX, "7_metadata_HSI-BRCA.xlsx"),
        file_type="xlsx",
        output_dir="outputs", dataset_name="HSIBRCA",
        exclude_columns=["Project Short Name"], id_columns=["Case ID"],
        force_categorical=["Menopausal_status", "Dx_surgery", "Tumor_histologic_grade", "LVI", "PNI", "T", "N", "M", "ER", "HER2", "PR", "KI67", "Molecular_subtype"],
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # 8. SOPHIE
    AnalysisConfig(
        file_path=os.path.join(DATA_XLSX, "8_metadata_SOPHIE.xlsx"),
        file_type="xlsx",
        output_dir="outputs", dataset_name="SOPHIE",
        exclude_columns=["ID"], id_columns=["ID"],
        force_categorical=["Gender"],
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # 9. (HISTAI completo omitido — solo se usan sub-cohortes)
    # histai_breast_regroup
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "histai_breast_metadata_regroup.csv"),
        output_dir="outputs", dataset_name="HISTAI_BREAST",
        exclude_columns=["conclusion", "case_mapping", "icd10"],
        force_numeric=["age"],
        force_categorical=["diagnosis", "diff_diagnostic", "grossing", "micro_protocol", "gender", "specialization"],
        column_title_map={"age": "Edad (años)"},
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # histai_colorectal_b1_regroup
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "histai_colorectal_b1_metadata_regroup.csv"),
        output_dir="outputs", dataset_name="HISTAI_CRC_B1",
        exclude_columns=["conclusion", "case_mapping", "icd10"],
        force_numeric=["age"],
        force_categorical=["diagnosis", "diff_diagnostic", "grossing", "micro_protocol", "gender", "specialization"],
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
    # histai_colorectal_b2
    AnalysisConfig(
        file_path=os.path.join(DATA_CSV, "histai_colorectal_b2_metadata_regroup.csv"),
        output_dir="outputs", dataset_name="HISTAI_CRC_B2",
        exclude_columns=["conclusion", "case_mapping", "icd10"],
        force_numeric=["age"],
        force_categorical=["diagnosis", "diff_diagnostic", "grossing", "micro_protocol", "gender", "specialization"],
        save_reports=False, save_cleaned_data=False, save_plots=True, show_plots=False
    ),
]

IMAGES_DRIVE = os.environ.get("IMAGES_DRIVE", os.path.join(ROOT, "imagesDrive"))
print(f"Output base dir: {IMAGES_DRIVE}")
os.makedirs(IMAGES_DRIVE, exist_ok=True)

for cfg in configs:
    print(f"\n{'=' * 60}")
    print(f"Analizando: {cfg.dataset_name}")
    print(f"Archivo: {cfg.file_path}")
    try:
        analyzer = DatasetAnalyzer(cfg)
        reports = analyzer.run()
        print(f"  OK - {cfg.dataset_name}")
    except Exception as e:
        print(f"  ERROR - {cfg.dataset_name}: {e}")

print(f"\n{'=' * 60}")
print("¡Todos los análisis completados!")
print(f"Imágenes generadas en: {IMAGES_DRIVE}")
