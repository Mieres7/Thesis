# Reproducibilidad - Tesis: Evaluación de Foundation Models en Patología Digital

Código de la tesis de título: extracción de patches, generación de embeddings y evaluación de *foundation models* (UNI2, Virchow2, Phikon-v2, Prov-GigaPath) en tareas de clasificación de cáncer de mama y colorrectal.

## Estructura

```
├── data_prep/                                   # Preparación de datos
│   ├── build_filtered_configs.py                # Genera configs filtrados por metadata
│   ├── build_fused_surgen_metadata.py           # Fusión metadata SurGen (site/side)
│   └── download_models.py                       # Descarga de los 4 foundation models
├── extraction/
│   ├── patch_extractor.py                       # Extracción de patches desde WSI a HDF5
│   └── patch_extractor_opt.py                   # Versión optimizada (máscara de tejido GPU)
├── embeddings/
│   ├── embed_from_patches_opt.py                # Embeddings desde patches HDF5 (multi-GPU)
│   ├── get_embeddings.py / get_embeddings_opt.py# Embeddings directo desde WSI
├── patches_explorer/                            # Notebooks de exploración/calibración de filtros
├── config_files/                                # Configs YAML (patches y embeddings por dataset)
├── metadata_clean/                              # Metadata limpia de los 7 datasets
├── logs/
│   ├── patches/                                 # Logs de extracción de patches por dataset
│   └── embeddings/                              # Logs de generación de embeddings por dataset
├── eval/                                        # Evaluación (probes lineales/MLP, CV, bootstrap, fairness, OOD, center effect)
│   ├── config/                                  # Definición de datasets, tareas y modelos
│   ├── datasets/                                # Loaders de metadata y embeddings
│   ├── evaluation/                              # Split, métricas, bootstrap, fairness, cross-dataset, center effect
│   ├── models/                                  # Probes lineales y MLP
│   └── sandbox/                                 # Orquestadores del pipeline de evaluación
└── requirements.txt
```

## Pipeline

### 1. Preparación

```bash
# Descargar los foundation models (UNI2, Virchow2, Phikon-v2, Prov-GigaPath)
python data_prep/download_models.py

# Fusionar metadata SurGen (cohortes SR386 + SR1482)
python data_prep/build_fused_surgen_metadata.py

# Generar configs YAML filtrados por metadata de cada dataset
python data_prep/build_filtered_configs.py
```

### 2. Extracción de patches

```bash
python extraction/patch_extractor_opt.py config_files/patches_<dataset>_filtered.yaml
```

Genera HDF5 con `patches` + `coords` y `extraction_stats.csv` (filtro de tejido vía máscara GPU + filtro de saturación). Los logs quedan en `logs/patches/`.

### 3. Embeddings

```bash
python embeddings/embed_from_patches_opt.py config_files/embeddings_<dataset>_filtered.yaml
# o alternativas: embeddings/get_embeddings.py / get_embeddings_opt.py (directo desde WSI)
```

Genera HDF5 con `embeddings` + `coords`, normalización en GPU y autocast bf16. Los logs quedan en `logs/embeddings/`.

### 4. Evaluación

```bash
cd eval/sandbox
python run_prepare.py --all --tag <munay|prime>   # Construye pkl por servidor de embeddings
python run_merge.py                               # Merge de pkl finales por modelo/tarea
python run_eval.py --all --cross-dataset          # CV 5-fold / split fijo + bootstrap + fairness + OOD
# o todo en uno:
python run_all.py --step all --cross-dataset
# o pipeline completo SurGen fusionado:
bash fusion_surgen.sh
```

La configuración de tareas, modelos y split (CV vs. fijo) se centraliza en `eval/config/datasets.py`.

### 5. Figuras del capítulo 6

Los notebooks de `patches_explorer/` permiten calibrar los filtros de extracción (tejido, saturación) y visualizar patches por dataset. Las figuras y tablas del capítulo 6 se generan con los scripts de `tareas_cap6/` del repositorio de escritura de la tesis.

## Dependencias

Ver `requirements.txt`. Entorno típico: Python 3.10+, torch con CUDA, h5py, pandas, scikit-learn, aicspylibczi.

## Notas

- Ejecutar los scripts desde la raíz del repo: los scripts de `embeddings/` importan módulos de `extraction/` vía `sys.path`.
- Las rutas por defecto a WSIs, modelos y embeddings apuntan al servidor donde se ejecutó el pipeline (`/home/...`); se sobreescriben vía los YAML de `config_files/` o con `--config`.
- Los HDF5 de patches/embeddings, los pkl de resultados y las figuras generadas no se versionan (ver `.gitignore`).