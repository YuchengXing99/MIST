#!/usr/bin/env python3
"""
Multi-GPU BRCA survival sweep.

Iterates over configs × folds × seeds [× n_proto for MIST configs].
MIST configs expect ST prototype files named {proto_prefix}{N}.npy in --proto-dir.
Results are written to:
  <results-dir-root>/k{N}/<model>/seed_<S>/fold_<F>/   (MIST)
  <results-dir-root>/<model>/seed_<S>/fold_<F>/          (base)

Usage:
  python scripts/survival/run_brca.py \\
      --gpus 0 1 2 3 --jobs-per-gpu 1 \\
      --configs abmil abmil_mist \\
      --n-proto 4 8 16 32 64 \\
      --proto-dir /path/to/protos \\
      --data-root /path/to/TCGA-BRCA \\
      --seeds 1 2 3 4 5 \\
      --results-dir-root results_survival

Dry-run to preview commands without launching:
  python scripts/survival/run_brca.py --dry-run ...
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


MIST_CONFIGS = {
    "abmil_mist", "transmil_mist", "ilra_mist",
    "meanmil_mist", "maxmil_mist", "clam_mist", "transformer_mil_mist", "dsmil_mist",
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

DEFAULT_SEEDS  = list(range(1, 11))
DEFAULT_FOLDS  = [0, 1, 2, 3, 4]
DEFAULT_N_PROTO = [16]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    config:     str
    fold:       int
    seed:       int
    n_proto:    int | None = None   # None for base (non-MIST) configs
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
        description="Multi-GPU BRCA survival sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpus", nargs="+", default=["0"],
                   help="GPU ids to use.")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Concurrent jobs per GPU.")
    p.add_argument("--configs", nargs="+", choices=ALL_CONFIGS, default=ALL_CONFIGS,
                   help="Model configs to run.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS)

    # Prototype settings
    p.add_argument("--n-proto", nargs="+", type=int, default=DEFAULT_N_PROTO,
                   help="Prototype counts to sweep (MIST configs only). "
                        "E.g. --n-proto 4 8 16 32 64.")
    p.add_argument("--proto-dir", default=None,
                   help="Directory containing prototype files named "
                        "{proto-prefix}{N}.npy (e.g. prototypes_k16.npy).")
    p.add_argument("--proto-prefix", default="prototypes_k",
                   help="Filename prefix for prototype files inside --proto-dir.")

    # Data / task
    p.add_argument("--data-root", dest="data_roots", action="append", default=None,
                   help="BRCA feature root. Repeat for multiple roots.")
    p.add_argument("--task",        default="BRCA_survival")
    p.add_argument("--target-col",  default="dss_survival_days")
    p.add_argument("--split-names", default="train,test")

    # Output / logging
    p.add_argument("--results-dir-root", default="results_survival")
    p.add_argument("--log-dir",    default=None,
                   help="Directory for per-job log files.")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without launching.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for config in args.configs:
        is_mist = config in MIST_CONFIGS
        proto_n_list = args.n_proto if is_mist else [None]

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

            for fold in args.folds:
                for seed in args.seeds:
                    jobs.append(Job(
                        config=config, fold=fold, seed=seed,
                        n_proto=n, proto_path=proto_path,
                    ))
    return jobs


# ---------------------------------------------------------------------------
# Command / env construction
# ---------------------------------------------------------------------------

def job_results_root(job: Job, base: str) -> str:
    """Results root for this job: base/k{N}/ for MIST, base/ for base models."""
    if job.n_proto is not None:
        return os.path.join(base, f"k{job.n_proto}")
    return base


def build_command_and_env(
    job: Job,
    src_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, str]]:
    command = [
        "bash",
        f"./scripts/survival/{job.config}.sh",
        "__GPU__",           # replaced at launch time
        args.task,
        args.target_col,
        f"survival/TCGA_BRCA_overall_survival_k={job.fold}",
        args.split_names,
    ]
    data_roots = args.data_roots or ["/path/to/TCGA-BRCA"]
    command.extend(data_roots)

    env: dict[str, str] = {
        "SEED_LIST":         str(job.seed),
        "RESULTS_DIR_ROOT":  job_results_root(job, args.results_dir_root),
        "OMP_NUM_THREADS":   "1",
        "MKL_NUM_THREADS":   "1",
        "NUMEXPR_NUM_THREADS": "1",
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
    job: Job,
    slot: GPUSlot,
    src_dir: Path,
    args: argparse.Namespace,
    log_dir: Path,
) -> RunningJob:
    command, extra_env = build_command_and_env(job, src_dir, args)
    command[2] = slot.gpu      # fill in GPU id

    suffix = f"__k{job.n_proto}" if job.n_proto is not None else ""
    log_name = f"{job.config}{suffix}__fold{job.fold}__seed{job.seed}__gpu{slot.gpu}.log"
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
    return f"{job.config}{suffix} fold={job.fold} seed={job.seed}"


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
    src_dir = Path(__file__).resolve().parents[2]
    log_dir = (
        Path(args.log_dir).expanduser().resolve() if args.log_dir
        else src_dir.parent / "logs" / "brca_survival" / time.strftime("%Y%m%d_%H%M%S")
    )

    jobs_list = build_jobs(args)
    jobs: deque[Job] = deque(jobs_list)
    slots: deque[GPUSlot] = deque(
        GPUSlot(gpu=gpu, slot_idx=i)
        for i in range(max(1, args.jobs_per_gpu))
        for gpu in args.gpus
    )
    running: dict[str, RunningJob] = {}
    failures: list[str] = []

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
            # Fill available slots
            while jobs and slots:
                slot = slots.popleft()
                job  = jobs.popleft()
                r    = launch_job(job, slot, src_dir, args, log_dir)
                key  = f"{slot.gpu}:{slot.slot_idx}"
                running[key] = r
                print(f"[START] gpu={slot.gpu}  {job_label(job)}  log={r.log_path.name}")

            # Reap finished jobs
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

    print("All BRCA survival jobs finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
