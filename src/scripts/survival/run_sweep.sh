#!/bin/bash
# Unified survival sweep launcher.
#
# Usage:
#   DATASET=luad DATA_ROOTS="/data/TCGA-LUAD /data/CPTAC-LUAD /data/NLST" \
#   ./scripts/survival/run_sweep.sh [gpu_id ...]
#
# Required env:
#   DATASET         brca / luad / lusc
#
# Optional env:
#   CONFIGS         space-separated model configs
#   SEEDS           space-separated seed list      (default: 1..10)
#   FOLDS           space-separated fold list      (default: 0 1 2 3 4)
#   N_PROTO         prototype counts for MIST      (default: 16)
#   PROTO_DIR       dir with prototypes_k{N}.npy
#   DATA_ROOTS      space-separated feature roots
#   RESULTS_DIR_ROOT results root                  (default: results_survival)
#   JOBS_PER_GPU    concurrent jobs per GPU        (default: 1)
#   LOG_DIR         log directory

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_dir="$(cd "${script_dir}/../.." && pwd)"

: "${DATASET:?DATASET env var is required (brca / luad / lusc)}"

if [[ $# -gt 0 ]]; then
    gpus=("$@")
else
    gpus=(0)
fi

read -r -a n_proto  <<< "${N_PROTO:-16}"
read -r -a seeds    <<< "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
read -r -a folds    <<< "${FOLDS:-0 1 2 3 4}"

results_dir_root="${RESULTS_DIR_ROOT:-results_survival}"
jobs_per_gpu="${JOBS_PER_GPU:-1}"
proto_dir="${PROTO_DIR:-}"
log_dir="${LOG_DIR:-}"

cmd=(
    python
    "./scripts/survival/run_sweep.py"
    --dataset    "${DATASET}"
    --gpus       "${gpus[@]}"
    --jobs-per-gpu "${jobs_per_gpu}"
    --n-proto    "${n_proto[@]}"
    --seeds      "${seeds[@]}"
    --folds      "${folds[@]}"
    --results-dir-root "${results_dir_root}"
)

# Optional: configs override
if [[ -n "${CONFIGS:-}" ]]; then
    read -r -a config_arr <<< "${CONFIGS}"
    cmd+=(--configs "${config_arr[@]}")
fi

[[ -n "${proto_dir}" ]] && cmd+=(--proto-dir "${proto_dir}")
[[ -n "${log_dir}" ]]   && cmd+=(--log-dir   "${log_dir}")

if [[ -n "${DATA_ROOTS:-}" ]]; then
    read -r -a data_roots <<< "${DATA_ROOTS}"
    for r in "${data_roots[@]}"; do
        cmd+=(--data-root "${r}")
    done
fi

(
    cd "${src_dir}" || exit 1
    "${cmd[@]}"
)
