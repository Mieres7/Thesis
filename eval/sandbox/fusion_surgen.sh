#!/usr/bin/env bash
# ============================================================================
# fusion_surgen.sh — Pipeline completo para SurGen fusionado (SR1482+SR386)
# ----------------------------------------------------------------------------
# PREPARACIÓN LOCAL (ya hecha, se sincroniza con `git pull` en el servidor):
#   - build_fused_surgen_metadata.py  -> SURGEN_CRC_site.csv (841) / _side.csv (707)
#   - eval/config/datasets.py         -> specs SurGen únicos + TASKS + EVAL_CONFIG
#   - eval/datasets/embedding_loader.py -> ids "SR1482_106" (prefijo+caso)
#   - eval/evaluation/fairness.py / verify_results.py arreglados
#
# EN EL SERVIDOR (desde ~/tesis):
#   git pull
#   bash fusion_surgen.sh
#
# FLUJO CON DOS SERVIDORES (munay + prime):
#   1) En PRIME (donde están HISTAI_Breast / HISTAI_CRC_B1): preparar los
#      .pkl parciales y sincronizarlos vía git:
#        cd eval/sandbox && python run_prepare.py --all --tag prime
#        git add prime/ && git commit -m "pkl prime" && git push
#   2) En MUNAY (donde está el resto de los embeddings):
#        git pull   # trae eval/sandbox/prime/*.pkl
#        bash fusion_surgen.sh   # prepara munay, MERGE de ambos tags, eval
#   3) Para las tareas de BREAST (no entran en TASKS_CRC), correr aparte:
#        python run_prepare.py --all --tag munay
#        python run_merge.py --input-dir . --output-dir "$PKL_DIR"
#        python run_eval.py --all --pkl-dir "$PKL_DIR" \
#            --output-dir "$OUT_DIR" --cross-dataset
#   4) Verificar:
#        python check_pkls.py --pkl-dir "$PKL_DIR"
#        python verify_results.py --output-dir "$OUT_DIR" --pkl-dir "$PKL_DIR"
#        python verify_results.py --output-dir "$OUT_DIR" --pkl-dir "$PKL_DIR" --probe mlp
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Config ────────────────────────────────────────────────────────────────
SRV_TAG="${SRV_TAG:-munay}"                 # tag del servidor (munay|prime)
OUT_DIR="${OUT_DIR:-results_v2_cv_universal}"
NPZ_DIR="${NPZ_DIR:-npz_v2_cv_universal}"
PKL_DIR="${PKL_DIR:-pkl_final}"
MODELS=("uni2" "virchow2" "phikon_v2" "prov_gigapath")
TASKS_CRC=("crc_site" "crc_side" "organ")   # solo tareas que tocan SurGen

cd "$SCRIPT_DIR"

echo "━━━ [1/4] Generar metadata fusionada (idempotente) ━━━"
python "$REPO_ROOT/data_prep/build_fused_surgen_metadata.py"

echo "━━━ [2/4] run_prepare + run_merge ──────────────────"
# Nota: los .pkl viejos de crc (con SurGen_1482/386) NO se borran:
# run_merge los SOBREESCRIBE al regenerar el pkl con "SurGen" fusionado.
for model in "${MODELS[@]}"; do
  for task in "${TASKS_CRC[@]}"; do
    python run_prepare.py --task "$task" --model "$model" --tag "$SRV_TAG"
  done
done
# Merge de TODOS los tags con .pkl presentes (munay + prime). NO filtrar
# por --tags "$SRV_TAG": si se filtra, pkl_final queda solo con lo local y
# HISTAI_Breast / HISTAI_CRC_B1 (preparados en prime) vuelven a faltar.
# Los tags parciales se sincronizan entre servidores vía git (eval/sandbox/{tag}/).
python run_merge.py --input-dir . --output-dir "$PKL_DIR"

echo "━━━ [3/4] run_eval (crc_site, crc_side, organ) ─────"
for model in "${MODELS[@]}"; do
  for task in "${TASKS_CRC[@]}"; do
    python run_eval.py --task "$task" --model "$model" \
        --pkl-dir "$PKL_DIR" --output-dir "$OUT_DIR" --cross-dataset --no-npz
  done
done

echo "━━━ [3.5/4] Limpiar artefactos viejos de SurGen_1482/SurGen_386 ──"
rm -f "$OUT_DIR"/*_SurGen_1482_* "$OUT_DIR"/*_SurGen_1482.*
rm -f "$OUT_DIR"/*_SurGen_386_*  "$OUT_DIR"/*_SurGen_386.*
rm -f "$NPZ_DIR"/*_SurGen_1482_* "$NPZ_DIR"/*_SurGen_1482.*
rm -f "$NPZ_DIR"/*_SurGen_386_*  "$NPZ_DIR"/*_SurGen_386.*

echo "━━━ [5/4] Verificación ─────────────────────────────"
python verify_results.py --output-dir "$OUT_DIR" --pkl-dir "$PKL_DIR"
python verify_results.py --output-dir "$OUT_DIR" --pkl-dir "$PKL_DIR" --probe mlp

echo "✔ Listo. SurGen ahora corre como un único dataset en $OUT_DIR"