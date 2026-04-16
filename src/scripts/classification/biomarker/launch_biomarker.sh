#!/bin/bash
# =============================================================================
#  MAMMOTH Biomarker Benchmark — MIST reproduction
#  arXiv:2603.22198 "Mixture of Mini Experts: Overcoming the Linear Layer
#  Bottleneck in Multiple Instance Learning"
#
#  Runs all 13 biomarker tasks across 4 datasets (TCGA-Lung, TCGA-BRCA,
#  TCGA-GBMLGG, BCNB) with baseline + MIST variants.
#
#  ---------------------------------------------------------------------------
#  To launch only a subset of datasets, comment out the blocks you don't want.
#  Edit the DATA_ROOTS and PROTO_DIR paths below to point at your feature dirs
#  and per-organ prototype files.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
TCGA_LUAD_ROOT="/path/to/TCGA-LUAD"
TCGA_LUSC_ROOT="/path/to/TCGA-LUSC"
TCGA_BRCA_ROOT="/path/to/TCGA-BRCA"
TCGA_GBM_ROOT="/path/to/TCGA-GBM"
TCGA_LGG_ROOT="/path/to/TCGA-LGG"
BCNB_ROOT="/path/to/BCNB"
BCNB_PROTO_DIR="/path/to/mol_vis_by_organ/Breast/prototypes_by_k"

LUNG_PROTO_DIR="/path/to/mol_vis_by_organ/Lung/prototypes_by_k"
BRCA_PROTO_DIR="/path/to/mol_vis_by_organ/Breast/prototypes_by_k"
GBMLGG_PROTO_DIR="/path/to/mol_vis_by_organ/Brain/prototypes_by_k"

RESULTS_DIR_ROOT="results_biomarker"
GPUS="0 1 2 3 4 5 6 7"
JOBS_PER_GPU=2
SEEDS="1 2 3 4 5"
# ====================================================================

cd "${SRC_DIR}"

# MAMMOTH Appendix A2: "cross-entropy loss with random class-weighted sampling"
export TRAIN_SAMPLER=weighted

BASELINES="abmil transmil ilra meanmil maxmil clam transformer_mil dsmil"
MIST_VARIANTS="abmil_mist transmil_mist ilra_mist meanmil_mist maxmil_mist clam_mist transformer_mil_mist dsmil_mist"

run_sweep() {
    local dataset=$1; shift
    python scripts/classification/biomarker/run_sweep.py \
        --dataset "${dataset}" \
        --gpus ${GPUS} \
        --jobs-per-gpu "${JOBS_PER_GPU}" \
        --seeds ${SEEDS} \
        --skip-existing \
        --results-dir-root "${RESULTS_DIR_ROOT}" \
        "$@"
}

# --- TCGA-Lung: TP53, KRAS, STK11, EGFR ---
echo "===== TCGA-Lung Phase 1: Baselines ====="
run_sweep tcga_lung --configs ${BASELINES} \
    --data-root "${TCGA_LUAD_ROOT}" --data-root "${TCGA_LUSC_ROOT}" \
    || echo "WARN: tcga_lung baselines had failures."
echo "===== TCGA-Lung Phase 2: MIST ====="
run_sweep tcga_lung --configs ${MIST_VARIANTS} --n-proto 16 \
    --proto-dir "${LUNG_PROTO_DIR}" \
    --data-root "${TCGA_LUAD_ROOT}" --data-root "${TCGA_LUSC_ROOT}" \
    || echo "WARN: tcga_lung MIST had failures."

# --- TCGA-BRCA: HER2, ER, PR, PIK3CA ---
echo "===== TCGA-BRCA Phase 1: Baselines ====="
run_sweep tcga_brca --configs ${BASELINES} \
    --data-root "${TCGA_BRCA_ROOT}" \
    || echo "WARN: tcga_brca baselines had failures."
echo "===== TCGA-BRCA Phase 2: MIST ====="
run_sweep tcga_brca --configs ${MIST_VARIANTS} --n-proto 16 \
    --proto-dir "${BRCA_PROTO_DIR}" \
    --data-root "${TCGA_BRCA_ROOT}" \
    || echo "WARN: tcga_brca MIST had failures."

# --- TCGA-GBMLGG: IDH1, histomol subtype ---
echo "===== TCGA-GBMLGG Phase 1: Baselines ====="
run_sweep tcga_gbmlgg --configs ${BASELINES} \
    --data-root "${TCGA_GBM_ROOT}" --data-root "${TCGA_LGG_ROOT}" \
    || echo "WARN: tcga_gbmlgg baselines had failures."
echo "===== TCGA-GBMLGG Phase 2: MIST ====="
run_sweep tcga_gbmlgg --configs ${MIST_VARIANTS} --n-proto 16 \
    --proto-dir "${GBMLGG_PROTO_DIR}" \
    --data-root "${TCGA_GBM_ROOT}" --data-root "${TCGA_LGG_ROOT}" \
    || echo "WARN: tcga_gbmlgg MIST had failures."

# --- BCNB: ER, PR, HER2 (10-fold CV) ---
echo "===== BCNB Phase 1: Baselines ====="
run_sweep bcnb --configs ${BASELINES} \
    --data-root "${BCNB_ROOT}" \
    || echo "WARN: bcnb baselines had failures."
echo "===== BCNB Phase 2: MIST ====="
run_sweep bcnb --configs ${MIST_VARIANTS} --n-proto 16 \
    --proto-dir "${BCNB_PROTO_DIR}" \
    --data-root "${BCNB_ROOT}" \
    || echo "WARN: bcnb MIST had failures."

echo "===== All biomarker experiments finished! ====="
