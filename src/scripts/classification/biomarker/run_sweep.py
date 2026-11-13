#!/usr/bin/env python3
"""
Multi-GPU biomarker classification sweep for the MAMMOTH benchmark
(arXiv:2603.22198, "Mixture of Mini Experts: Overcoming the Linear Layer
Bottleneck in Multiple Instance Learning").

Reproduces the paper's 13 biomarker prediction tasks across 4 datasets
(TCGA-Lung, TCGA-BRCA, TCGA-GBMLGG, BCNB) with MIST + baseline MIL variants.

Iterates over tasks x folds x seeds x models. For MIST configs you can also
sweep n_proto values and supply per-n_proto prototype .npy files.

Usage (single dataset, all models):
  python scripts/classification/biomarker/run_sweep.py \\
      --dataset tcga_lung \\
      --gpus 0 1 2 3 \\
      --jobs-per-gpu 2 \\
      --configs abmil abmil_mist transmil transmil_mist ilra ilra_mist \\
      --seeds 1 2 3 4 5 --folds 0 1 2 3 4 \\
      --data-root /data/TCGA-LUAD --data-root /data/TCGA-LUSC \\
      --proto-dir /data/prototypes/Lung/prototypes_by_k --n-proto 16

Dry-run to preview commands:
  python scripts/classification/biomarker/run_sweep.py --dataset tcga_brca --dry-run ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Dataset + task presets (MAMMOTH biomarker benchmark)
# ---------------------------------------------------------------------------
# Each task is a tuple: (task_name, target_col).
# split_tpl is formatted with {task} and {fold}; every task of a dataset
# shares the same split-dir template.
#
# Task names must match the keys registered in src/data_factory/cls_default.py.

DATASET_PRESETS: dict[str, dict] = {
    "tcga_lung": {
        "tasks": [
            ("tcga_lung_tp53",  "tp53_mut"),
            ("tcga_lung_kras",  "kras_mut"),
            ("tcga_lung_stk11", "stk11_mut"),
            ("tcga_lung_egfr",  "egfr_mut"),
        ],
        "split_tpl":   "classification/biomarker/{task}_k={fold}",
        "split_names": "train,val,test",
        "log_subdir":  "biomarker_tcga_lung",
        "default_folds": [0, 1, 2, 3, 4],
        "placeholder_roots": ["/path/to/TCGA-LUAD", "/path/to/TCGA-LUSC"],
    },
    "tcga_brca": {
        "tasks": [
            ("tcga_brca_her2",   "her2_status"),
            ("tcga_brca_er",     "er_status"),
            ("tcga_brca_pr",     "pr_status"),
            ("tcga_brca_pik3ca", "pik3ca_mut"),
        ],
        "split_tpl":   "classification/biomarker/{task}_k={fold}",
        "split_names": "train,val,test",
        "log_subdir":  "biomarker_tcga_brca",
        "default_folds": [0, 1, 2, 3, 4],
        "placeholder_roots": ["/path/to/TCGA-BRCA"],
    },
    "tcga_gbmlgg": {
        "tasks": [
            ("tcga_gbmlgg_idh1",     "idh1_mut"),
            ("tcga_gbmlgg_histomol", "histomol_subtype"),
        ],
        "split_tpl":   "classification/biomarker/{task}_k={fold}",
        "split_names": "train,val,test",
        "log_subdir":  "biomarker_tcga_gbmlgg",
        "default_folds": [0, 1, 2, 3, 4],
        "placeholder_roots": ["/path/to/TCGA-GBM", "/path/to/TCGA-LGG"],
    },
    "bcnb": {
        "tasks": [
            ("bcnb_er",   "er_status"),
            ("bcnb_pr",   "pr_status"),
            ("bcnb_her2", "her2_status"),
        ],
        "split_tpl":   "classification/biomarker/{task}_k={fold}",
        "split_names": "train,val,test",
        "log_subdir":  "biomarker_bcnb",
        "default_folds": list(range(10)),
        "placeholder_roots": ["/path/to/BCNB"],
    },
}


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MIST_CONFIGS = {
    "abmil_mist", "transmil_mist", "ilra_mist",
    "meanmil_mist", "maxmil_mist", "clam_mist",
    "transformer_mil_mist", "dsmil_mist",
}
ALL_CONFIGS = [
    "abmil", "abmil_mist",
    "transmil", "transmil_mist",
    "ilra", "ilra_mist",
    "meanmil", "meanmil_mist",
    "maxmil", "maxmil_mist",
    "clam", "clam_mist",
    "transformer_mil", "transformer_mil_mist",
    "dsmil", "dsmil_mist",
]

# Result dir = model_config.lower().replace('_default', '') in main_classification.
# model_config = "{Model}_{suffix}" where suffix ∈ {default, MIST_default}.
CONFIG_TO_RESULT_DIR = {
    "abmil":            "abmil",
    "abmil_mist":       "abmil_mist",
    "transmil":         "transmil",
    "transmil_mist":    "transmil_mist",
    "ilra":             "ilra",
    "ilra_mist":        "ilra_mist",
    "meanmil":          "meanmil",
    "meanmil_mist":     "meanmil_mist",
    "maxmil":           "maxmil",
    "maxmil_mist":      "maxmil_mist",
    "clam":             "clam",
    "clam_mist":        "clam_mist",
    "transformer_mil":      "transformermil",
    "transformer_mil_mist": "transformermil_mist",
    "dsmil":            "dsmil",
    "dsmil_mist":       "dsmil_mist",
}

DEFAULT_SEEDS   = [1, 2, 3, 4, 5]
DEFAULT_N_PROTO = [16]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    config:     str
    task:       str
    target_col: str
    fold:       int
    seed:       int
    n_proto:    int | None = None
    proto_path: str | None = None


@dataclass
class RunningJob:
    job:        Job
    gpu:        str
    slot_idx:   int
    proc:       subprocess.Popen
    log_path:   Path
    log_handle: object
    started_at: float


class GPUSlot(NamedTuple):
    gpu:      str
    slot_idx: int


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-GPU biomarker classification sweep (MAMMOTH benchmark).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()), required=True,
                   help="Dataset preset (tcga_lung / tcga_brca / tcga_gbmlgg / bcnb).")

    # Task subset (within a dataset preset). If omitted, runs all tasks for the dataset.
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Subset of task names (must be keys in the dataset preset).")

    # GPU / parallelism
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--jobs-per-gpu", type=int, default=1)

    # Model selection
    p.add_argument("--configs", nargs="+", choices=ALL_CONFIGS, default=ALL_CONFIGS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="Fold indices. If omitted, uses dataset preset default.")

    # Prototype settings (MIST only)
    p.add_argument("--n-proto", nargs="+", type=int, default=DEFAULT_N_PROTO)
    p.add_argument("--proto-dir", default=None,
                   help="Dir containing prototypes_k<N>.npy files.")
    p.add_argument("--proto-prefix", default="prototypes_k")

    # Data / split overrides
    p.add_argument("--data-root", dest="data_roots", action="append", default=None,
                   help="Feature root dir (repeat for multiple).")
    p.add_argument("--split-names", default=None)
    p.add_argument("--split-tpl", default=None,
                   help="Override split-dir template. {task} / {fold} are substituted.")

    # Output / logging
    p.add_argument("--results-dir-root", default="results_biomarker")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip jobs whose summary.csv already exists.")

    args = p.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    args.preset_tasks        = preset["tasks"]
    args.split_tpl           = args.split_tpl or preset["split_tpl"]
    args.split_names         = args.split_names or preset["split_names"]
    args.log_subdir          = preset["log_subdir"]
    args.placeholder_roots   = preset["placeholder_roots"]
    if args.folds is None:
        args.folds = list(preset["default_folds"])

    # Filter tasks if subset requested
    if args.tasks:
        valid_names = {t[0] for t in preset["tasks"]}
        unknown = set(args.tasks) - valid_names
        if unknown:
            p.error(f"Unknown tasks for dataset {args.dataset}: {sorted(unknown)}. "
                    f"Valid: {sorted(valid_names)}")
        args.preset_tasks = [t for t in preset["tasks"] if t[0] in args.tasks]

    return args


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for config in args.configs:
        is_mist = config in MIST_CONFIGS
        proto_n_list: list[int | None] = list(args.n_proto) if is_mist else [None]

        for n in proto_n_list:
            proto_path = None
            if is_mist and args.proto_dir and n is not None:
                candidate = Path(args.proto_dir) / f"{args.proto_prefix}{n}.npy"
                if candidate.exists():
                    proto_path = str(candidate)
                else:
                    print(f"[WARN] Proto file not found: {candidate} "
                          f"— skipping n_proto={n} for {config}.")
                    continue

            for task_name, target_col in args.preset_tasks:
                for fold in args.folds:
                    for seed in args.seeds:
                        jobs.append(Job(
                            config=config, task=task_name, target_col=target_col,
                            fold=fold, seed=seed, n_proto=n, proto_path=proto_path,
                        ))
    return jobs


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def job_results_root(job: Job, base: str) -> str:
    """
    Layout: <results_root>[/k<N>]/<task>/k=<fold>/
    main_classification will further nest <model>/seed_<seed>/ inside this.
    Uses forward slashes so the path is shell-safe on Linux.
    """
    parts = [base.rstrip("/\\")]
    if job.n_proto is not None:
        parts.append(f"k{job.n_proto}")
    parts.append(job.task)
    parts.append(f"k={job.fold}")
    return "/".join(parts)


def filter_completed_jobs(
    jobs: list[Job], results_dir_root: str, src_dir: Path,
) -> list[Job]:
    repo_root = src_dir.parent
    remaining: list[Job] = []
    skipped = 0
    for job in jobs:
        base = job_results_root(job, results_dir_root)
        if not os.path.isabs(base):
            base = os.path.join(str(repo_root), base)
        result_dir_name = CONFIG_TO_RESULT_DIR.get(job.config, job.config)
        summary_path = os.path.join(
            base, result_dir_name, f"seed_{job.seed}", "summary.csv"
        )
        if os.path.isfile(summary_path):
            skipped += 1
        else:
            remaining.append(job)
    if skipped:
        print(f"[skip-existing] Skipped {skipped} completed jobs, {len(remaining)} remaining.")
    return remaining


# ---------------------------------------------------------------------------
# Command / env construction
# ---------------------------------------------------------------------------

def build_command_and_env(
    job: Job, src_dir: Path, args: argparse.Namespace,
) -> tuple[list[str], dict[str, str]]:
    split_dir = args.split_tpl.format(task=job.task, fold=job.fold)
    command = [
        "bash",
        f"./scripts/classification/biomarker/models/{job.config}.sh",
        "__GPU__",
        job.task,
        job.target_col,
        split_dir,
        args.split_names,
    ]
    data_roots = args.data_roots or args.placeholder_roots
    command.extend(data_roots)

    env: dict[str, str] = {
        "SEED_LIST":           str(job.seed),
        "RESULTS_DIR_ROOT":    job_results_root(job, args.results_dir_root),
        "OMP_NUM_THREADS":     "1",
        "MKL_NUM_THREADS":     "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED":    "1",
    }
    if job.proto_path:
        env["ST_PROTO_PATH"] = job.proto_path
    if job.n_proto is not None:
        env["N_PROTO"] = str(job.n_proto)

    return command, env


# ---------------------------------------------------------------------------
# Launching / scheduling
# ---------------------------------------------------------------------------

def launch_job(
    job: Job, slot: GPUSlot, src_dir: Path,
    args: argparse.Namespace, log_dir: Path,
) -> RunningJob:
    command, extra_env = build_command_and_env(job, src_dir, args)
    command[2] = slot.gpu

    proto_suffix = f"__k{job.n_proto}" if job.n_proto is not None else ""
    log_name = (
        f"{job.config}{proto_suffix}__{job.task}"
        f"__fold{job.fold}__seed{job.seed}__gpu{slot.gpu}.log"
    )
    log_path = log_dir / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    env.update(extra_env)

    proc = subprocess.Popen(
        command, cwd=str(src_dir), env=env,
        stdout=log_handle, stderr=subprocess.STDOUT,
    )
    return RunningJob(
        job=job, gpu=slot.gpu, slot_idx=slot.slot_idx,
        proc=proc, log_path=log_path, log_handle=log_handle,
        started_at=time.time(),
    )


def job_label(job: Job) -> str:
    suffix = f" k={job.n_proto}" if job.n_proto is not None else ""
    return f"{job.config}{suffix} {job.task} fold={job.fold} seed={job.seed}"


def terminate_all(running: dict[str, RunningJob]) -> None:
    for r in running.values():
        if r.proc.poll() is None:
            r.proc.terminate()
        r.log_handle.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    src_dir = Path(__file__).resolve().parents[3]
    log_dir = (
        Path(args.log_dir).expanduser().resolve() if args.log_dir
        else src_dir.parent / "logs" / args.log_subdir / time.strftime("%Y%m%d_%H%M%S")
    )

    jobs_list = build_jobs(args)
    if args.skip_existing:
        jobs_list = filter_completed_jobs(jobs_list, args.results_dir_root, src_dir)

    jobs: deque[Job] = deque(jobs_list)
    slots: deque[GPUSlot] = deque(
        GPUSlot(gpu=gpu, slot_idx=i)
        for i in range(max(1, args.jobs_per_gpu))
        for gpu in args.gpus
    )
    running: dict[str, RunningJob] = {}
    failures: list[str] = []

    print(f"Dataset       : {args.dataset}")
    print(f"Tasks         : {[t[0] for t in args.preset_tasks]}")
    print(f"Split names   : {args.split_names}")
    print(f"GPUs          : {' '.join(args.gpus)}  (jobs_per_gpu={args.jobs_per_gpu})")
    print(f"Configs       : {' '.join(args.configs)}")
    print(f"Seeds         : {args.seeds}")
    print(f"Folds         : {args.folds}")
    print(f"n_proto sweep : {args.n_proto}  proto_dir={args.proto_dir}")
    print(f"Results root  : {args.results_dir_root}")
    print(f"Total jobs    : {len(jobs)}")

    if args.dry_run:
        for job in jobs:
            cmd, env = build_command_and_env(job, src_dir, args)
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            print(f"  {env_str} {' '.join(cmd)}")
        return 0

    try:
        while jobs or running:
            while jobs and slots:
                slot = slots.popleft()
                job  = jobs.popleft()
                r    = launch_job(job, slot, src_dir, args, log_dir)
                key  = f"{slot.gpu}:{slot.slot_idx}"
                running[key] = r
                print(f"[START] gpu={slot.gpu}  {job_label(job)}  log={r.log_path.name}")

            done_keys: list[str] = []
            for key, r in running.items():
                rc = r.proc.poll()
                if rc is None:
                    continue
                r.log_handle.close()
                done_keys.append(key)
                slots.append(GPUSlot(r.gpu, r.slot_idx))
                label = job_label(r.job)
                if rc == 0:
                    print(f"[DONE]  gpu={r.gpu}  {label}")
                else:
                    msg = f"{label} rc={rc} log={r.log_path}"
                    failures.append(msg)
                    print(f"[FAIL]  {msg}")

            for key in done_keys:
                del running[key]

            if running and not done_keys:
                time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        print("\nInterrupted — terminating running jobs...", file=sys.stderr)
        terminate_all(running)
        return 1

    if failures:
        print("\nFailed jobs:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"All {args.dataset} biomarker jobs finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
