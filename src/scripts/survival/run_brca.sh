#!/bin/bash
# Launch BRCA survival sweep across multiple GPUs.
#
# Usage:
#   ./scripts/survival/run_brca.sh [gpu_id ...]
#
# Environment variables (all optional):
#   CONFIGS         space-separated model configs  (default: all 6)
#   SEEDS           space-separated seed list      (default: 1..10)
#   FOLDS           space-separated fold list      (default: 0 1 2 3 4)
#   N_PROTO         space-separated prototype counts for MIST configs
#                   (default: 16; e.g. "4 8 16 32 64")
#   PROTO_DIR       dir containing prototypes_k{N}.npy files
#   DATA_ROOT       single BRCA feature root
#   DATA_ROOTS      space-separated list of BRCA feature roots
#   RESULTS_DIR_ROOT results root dir              (default: results_survival)
#   JOBS_PER_GPU    concurrent jobs per GPU        (default: 1)
#   LOG_DIR         log directory
#
# Examples:
#   # Single GPU, all configs, default 16 prototypes
#   DATA_ROOT=/data/TCGA-BRCA ./scripts/survival/run_brca.sh 0
#
#   # 4 GPUs, MIST-only, sweep prototype counts
#   CONFIGS="abmil_mist transmil_mist ilra_mist" \
#   N_PROTO="4 8 16 32 64" \
#   PROTO_DIR=/data/protos \
#   DATA_ROOT=/data/TCGA-BRCA \
#   JOBS_PER_GPU=2 \
#   ./scripts/survival/run_brca.sh 0 1 2 3

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_dir="$(cd "${script_dir}/../.." && pwd)"

# GPU list from positional args (default: GPU 0)
if [[ $# -gt 0 ]]; then
    gpus=("$@")
else
    gpus=(0)
fi

# Configurable via env vars
read -r -a n_proto  <<< "${N_PROTO:-16}"
read -r -a seeds    <<< "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
read -r -a folds    <<< "${FOLDS:-0 1 2 3 4}"
read -r -a configs  <<< "${CONFIGS:-abmil abmil_mist transmil transmil_mist ilra ilra_mist}"

results_dir_root="${RESULTS_DIR_ROOT:-results_survival}"
jobs_per_gpu="${JOBS_PER_GPU:-1}"
proto_dir="${PROTO_DIR:-}"
log_dir="${LOG_DIR:-}"

cmd=(
    python
    "./scripts/survival/run_brca.py"
    --gpus       "${gpus[@]}"
    --jobs-per-gpu "${jobs_per_gpu}"
    --configs    "${configs[@]}"
    --n-proto    "${n_proto[@]}"
    --seeds      "${seeds[@]}"
    --folds      "${folds[@]}"
    --results-dir-root "${results_dir_root}"
)

[[ -n "${proto_dir}" ]]  && cmd+=(--proto-dir  "${proto_dir}")
[[ -n "${log_dir}" ]]    && cmd+=(--log-dir     "${log_dir}")

if [[ -n "${DATA_ROOT:-}" ]]; then
    cmd+=(--data-root "${DATA_ROOT}")
elif [[ -n "${DATA_ROOTS:-}" ]]; then
    read -r -a data_roots <<< "${DATA_ROOTS}"
    for r in "${data_roots[@]}"; do
        cmd+=(--data-root "${r}")
    done
fi

(
    cd "${src_dir}" || exit 1
    "${cmd[@]}"
)
