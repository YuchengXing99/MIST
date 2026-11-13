#!/bin/bash
# =============================================================================
#  LUAD + LUSC — Parallel Launch on 8x H200
#  LUAD → GPU 0-3 | LUSC → GPU 4-7
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ========================= EDIT THESE PATHS =========================
LUAD_DATA_ROOTS="/path/to/TCGA-LUAD /path/to/CPTAC-LUAD /path/to/NLST"
LUSC_DATA_ROOTS="/path/to/TCGA-LUSC /path/to/CPTAC-LUSC /path/to/NLST"
PROTO_DIR="/path/to/prototypes"
LUAD_RESULTS="results_survival_luad"
LUSC_RESULTS="results_survival_lusc"
# ====================================================================

SEEDS="1 2 3 4 5 6 7 8 9 10"
FOLDS="0 1 2 3 4"
N_PROTO="16"
JOBS_PER_GPU=2

echo "=========================================="
echo "  LUAD (GPU 0-3) + LUSC (GPU 4-7)"
echo "=========================================="

# LUAD on GPU 0-3
(
    DATASET=luad \
    DATA_ROOTS="${LUAD_DATA_ROOTS}" \
    SEEDS="${SEEDS}" FOLDS="${FOLDS}" \
    N_PROTO="${N_PROTO}" PROTO_DIR="${PROTO_DIR}" \
    RESULTS_DIR_ROOT="${LUAD_RESULTS}" \
    JOBS_PER_GPU="${JOBS_PER_GPU}" \
    bash "${SCRIPT_DIR}/run_sweep.sh" 0 1 2 3
) &
LUAD_PID=$!

# LUSC on GPU 4-7
(
    DATASET=lusc \
    DATA_ROOTS="${LUSC_DATA_ROOTS}" \
    SEEDS="${SEEDS}" FOLDS="${FOLDS}" \
    N_PROTO="${N_PROTO}" PROTO_DIR="${PROTO_DIR}" \
    RESULTS_DIR_ROOT="${LUSC_RESULTS}" \
    JOBS_PER_GPU="${JOBS_PER_GPU}" \
    bash "${SCRIPT_DIR}/run_sweep.sh" 4 5 6 7
) &
LUSC_PID=$!

echo "  LUAD PID: ${LUAD_PID} (GPU 0-3)"
echo "  LUSC PID: ${LUSC_PID} (GPU 4-7)"

FAIL=0
wait ${LUAD_PID} || { echo "[FAIL] LUAD"; FAIL=1; }
wait ${LUSC_PID} || { echo "[FAIL] LUSC"; FAIL=1; }

if [[ ${FAIL} -eq 0 ]]; then
    echo "===== All LUAD + LUSC experiments finished! ====="
else
    echo "===== Some experiments failed. Check logs. ====="
    exit 1
fi
