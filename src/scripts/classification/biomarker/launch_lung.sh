#!/bin/bash
# =============================================================================
#  TCGA-Lung Biomarker — Full Experiment Launch
#  Datasets: TCGA-LUAD + TCGA-LUSC (merged)
#  Tasks: TP53, KRAS, STK11, EGFR (5-fold CV)
#  8x H200 GPUs, 2 jobs per GPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
TCGA_LUAD_ROOT="/path/to/TCGA-LUAD"
TCGA_LUSC_ROOT="/path/to/TCGA-LUSC"
LUNG_PROTO_DIR="/path/to/mol_vis_by_organ/Lung/prototypes_by_k"
RESULTS_DIR_ROOT="results_biomarker_lung"
# ====================================================================

GPUS="0 1 2 3 4 5 6 7"
JOBS_PER_GPU=2
SEEDS="1 2 3 4 5"
FOLDS="0 1 2 3 4"
N_PROTO=16

BASELINES="abmil transmil ilra meanmil maxmil clam transformer_mil dsmil"
MIST_VARIANTS="abmil_mist transmil_mist ilra_mist meanmil_mist maxmil_mist clam_mist transformer_mil_mist dsmil_mist"

cd "${SRC_DIR}"

# MAMMOTH Appendix A2: "cross-entropy loss with random class-weighted sampling"
export TRAIN_SAMPLER=weighted

# --- Phase 1: Baselines ---
echo "===== TCGA-Lung Phase 1: Baselines ====="
python scripts/classification/biomarker/run_sweep.py \
    --dataset tcga_lung \
    --configs ${BASELINES} \
    --gpus ${GPUS} \
    --jobs-per-gpu "${JOBS_PER_GPU}" \
    --seeds ${SEEDS} \
    --folds ${FOLDS} \
    --skip-existing \
    --results-dir-root "${RESULTS_DIR_ROOT}" \
    --data-root "${TCGA_LUAD_ROOT}" \
    --data-root "${TCGA_LUSC_ROOT}" \
    || echo "WARN: tcga_lung baselines had failures."

# --- Phase 2: MIST variants ---
echo "===== TCGA-Lung Phase 2: MIST ====="
python scripts/classification/biomarker/run_sweep.py \
    --dataset tcga_lung \
    --configs ${MIST_VARIANTS} \
    --n-proto ${N_PROTO} \
    --proto-dir "${LUNG_PROTO_DIR}" \
    --gpus ${GPUS} \
    --jobs-per-gpu "${JOBS_PER_GPU}" \
    --seeds ${SEEDS} \
    --folds ${FOLDS} \
    --skip-existing \
    --results-dir-root "${RESULTS_DIR_ROOT}" \
    --data-root "${TCGA_LUAD_ROOT}" \
    --data-root "${TCGA_LUSC_ROOT}" \
    || echo "WARN: tcga_lung MIST had failures."

echo "===== All TCGA-Lung biomarker experiments finished! ====="
