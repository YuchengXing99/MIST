#!/bin/bash
# Shared runner for classification experiments. Mirrors _run_survival_model.sh.
#
# Callers (thin wrappers under biomarker/models/) set MODEL_TUPLE then source
# this file. Per-seed sweep jobs override SEED_LIST / RESULTS_DIR_ROOT /
# ST_PROTO_PATH / N_PROTO via environment variables.

set -euo pipefail

die() {
    echo "$1" >&2
    return 1 2>/dev/null || exit 1
}

if [[ -z "${MODEL_TUPLE:-}" ]]; then
    die "MODEL_TUPLE must be set before sourcing _run_classification_model.sh"
fi

if [[ $# -lt 6 ]]; then
    die "Usage: <gpu> <task> <target_col> <split_dir> <split_names> <data_root...>"
fi

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("${@:6}")
python_bin="${PYTHON_BIN:-python}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# _run_classification_model.sh lives at src/scripts/classification/, so src = ../..
src_dir="$(cd "${script_dir}/../.." && pwd)"
repo_root="$(cd "${src_dir}/.." && pwd)"

read -r -a seed_list <<< "${SEED_LIST:-1 2 3 4 5}"

feat="${FEAT_NAME:-extracted-vit_large_patch16_224.dinov2.uni_mass100k}"
input_dim="${INPUT_DIM:-1536}"
mag="${MAG:-20x}"
patch_size="${PATCH_SIZE:-256}"

max_epoch="${MAX_EPOCHS:-20}"
bag_size="${BAG_SIZE:--1}"
batch_size="${BATCH_SIZE:-1}"
lr="${LR:-0.0001}"
wd="${WD:-0.00001}"
opt="${OPT:-adamW}"
lr_scheduler="${LR_SCHEDULER:-cosine}"
num_workers="${NUM_WORKERS:-8}"
in_dropout="${IN_DROPOUT:-0.1}"
early_stopping="${EARLY_STOPPING:-1}"
save_dir_root="${RESULTS_DIR_ROOT:-results_biomarker}"
train_sampler="${TRAIN_SAMPLER:-random}"
st_proto_path="${ST_PROTO_PATH:-}"
n_proto="${N_PROTO:-}"
if [[ "${save_dir_root}" != /* ]]; then
    save_dir_root="${repo_root}/${save_dir_root}"
fi

IFS=',' read -r model config_suffix <<< "${MODEL_TUPLE}"
model_config="${model}_${config_suffix}"
feat_name="${feat#extracted-}"

th=0.00005
if awk "BEGIN {exit !($lr <= $th)}"; then
    warmup=0
    curr_lr_scheduler='constant'
else
    curr_lr_scheduler="${lr_scheduler}"
    warmup=1
fi

if [[ "${bag_size}" == "-1" ]]; then
    curr_bag_size="${bag_size}"
    curr_batch_size=1
    grad_accum=32
else
    if [[ "${patch_size}" == "512" ]]; then
        curr_bag_size='1024'
    else
        curr_bag_size="${bag_size}"
    fi
    curr_batch_size="${batch_size}"
    grad_accum=1
fi

all_feat_dirs=()
for dataroot_path in "${dataroots[@]}"; do
    feat_dir="${dataroot_path}/extracted_mag${mag}_patch${patch_size}_fp/${feat}/feats_h5"
    if [[ -d "${feat_dir}" ]]; then
        all_feat_dirs+=("${feat_dir}")
    fi
done

if [[ ${#all_feat_dirs[@]} -eq 0 ]]; then
    die "No feature directories found under the provided data roots for feat=${feat}, mag=${mag}, patch_size=${patch_size}."
fi

all_feat_dirs_joined=$(IFS=,; printf '%s' "${all_feat_dirs[*]}")

echo "Running ${model_config} on ${split_dir} (task=${task}, target=${target_col})"
echo "Seeds: ${seed_list[*]}"
echo "Feature dirs: ${all_feat_dirs_joined}"
echo "Results root: ${save_dir_root}"

for seed in "${seed_list[@]}"; do
    run_exp_code="${task}::${model_config}::${feat_name}::seed${seed}"
    cmd=(
        "${python_bin}" -m training.main_classification
        --data_source "${all_feat_dirs_joined}"
        --results_dir "${save_dir_root}"
        --split_dir "${split_dir}"
        --split_names "${split_names}"
        --task "${task}"
        --target_col "${target_col}"
        --model_type "${model}"
        --model_config "${model_config}"
        --in_dim "${input_dim}"
        --opt "${opt}"
        --lr "${lr}"
        --lr_scheduler "${curr_lr_scheduler}"
        --wd "${wd}"
        --warmup_epochs "${warmup}"
        --max_epochs "${max_epoch}"
        --train_bag_size "${curr_bag_size}"
        --batch_size "${curr_batch_size}"
        --accum_steps "${grad_accum}"
        --in_dropout "${in_dropout}"
        --seed "${seed}"
        --num_workers "${num_workers}"
        --train_sampler "${train_sampler}"
        --exp_code "${run_exp_code}"
    )
    if [[ "${early_stopping}" == "1" ]]; then
        cmd+=(--early_stopping)
    fi
    if [[ -n "${st_proto_path}" ]]; then
        cmd+=(--load_proto --proto_path "${st_proto_path}")
    fi
    if [[ -n "${n_proto}" ]]; then
        cmd+=(--n_proto "${n_proto}")
    fi

    echo "[START] model=${model_config} split=${split_dir} seed=${seed}"
    (
        cd "${src_dir}" || exit 1
        CUDA_VISIBLE_DEVICES="${gpuid}" "${cmd[@]}"
    )
    echo "[DONE] model=${model_config} split=${split_dir} seed=${seed}"
done
