#!/bin/bash
# =============================================================================
#  TCGA-GBMLGG Biomarker — Full Experiment Launch
#  Datasets: TCGA-GBM + TCGA-LGG (merged)
#  Tasks: IDH1 mutation (binary), histomolecular subtype (5-class) (5-fold CV)
#  8x H200 GPUs, 2 jobs per GPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
TCGA_GBM_ROOT="/hpc/home/e0950223/MIST_data/survival/TCGA-GBM"
TCGA_LGG_ROOT="/hpc/home/e0950223/MIST_data/survival/TCGA-LGG"
GBMLGG_PROTO_DIR="/hpc/home/e0950223/MIST_data/mol_vis_by_organ/Brain/prototypes_by_k"
RESULTS_DIR_ROOT="results_biomarker_gbmlgg"
# ====================================================================

GPUS="0 1 2 3 4 5"
JOBS_PER_GPU=2
SEEDS="1 2 3 4 5 6 7 8 9 10"
FOLDS="0 1 2 3 4"
N_PROTO=4

# MAMMOTH Appendix A2: "cross-entropy loss with random class-weighted sampling"
export TRAIN_SAMPLER=weighted

BASELINES="abmil transmil ilra meanmil maxmil clam transformer_mil dsmil"
MIST_VARIANTS="abmil_mist transmil_mist ilra_mist meanmil_mist maxmil_mist clam_mist transformer_mil_mist dsmil_mist"

cd "${SRC_DIR}"

# --- Phase 1: Baselines ---
echo "===== TCGA-GBMLGG Phase 1: Baselines ====="
python scripts/classification/biomarker/run_sweep.py \
    --dataset tcga_gbmlgg \
    --configs ${BASELINES} \
    --gpus ${GPUS} \
    --jobs-per-gpu "${JOBS_PER_GPU}" \
    --seeds ${SEEDS} \
    --folds ${FOLDS} \
    --skip-existing \
    --results-dir-root "${RESULTS_DIR_ROOT}" \
    --data-root "${TCGA_GBM_ROOT}" \
    --data-root "${TCGA_LGG_ROOT}" \
    || echo "WARN: tcga_gbmlgg baselines had failures."

# --- Phase 2: MIST variants ---
echo "===== TCGA-GBMLGG Phase 2: MIST ====="
python scripts/classification/biomarker/run_sweep.py \
    --dataset tcga_gbmlgg \
    --configs ${MIST_VARIANTS} \
    --n-proto ${N_PROTO} \
    --proto-dir "${GBMLGG_PROTO_DIR}" \
    --gpus ${GPUS} \
    --jobs-per-gpu "${JOBS_PER_GPU}" \
    --seeds ${SEEDS} \
    --folds ${FOLDS} \
    --skip-existing \
    --results-dir-root "${RESULTS_DIR_ROOT}" \
    --data-root "${TCGA_GBM_ROOT}" \
    --data-root "${TCGA_LGG_ROOT}" \
    || echo "WARN: tcga_gbmlgg MIST had failures."

echo "===== All TCGA-GBMLGG biomarker experiments finished! ====="
