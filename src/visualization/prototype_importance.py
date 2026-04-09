"""
Tool 3: Prototype Importance Analysis

Three analyses:
  1. Ablation — zero out each prototype, measure prediction change
  2. Slide-level profiles — mean affinity per slide → UMAP
  3. Cox regression (survival) — which prototypes predict outcome

Usage (from src/):
    python -m visualization.prototype_importance \
        --ckpt results/ABMIL_MIST/s_checkpoint.pth \
        --data-source /path/to/feats_h5 \
        --split-csv splits/task/test.csv \
        --out-dir viz_out/importance \
        --mode classification --label-col label
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from ._shared import (
    load_model_from_ckpt, extract_mist, load_h5, run_inference,
    collect_h5_paths, save_fig, ensure_dir, DPI, CMAP_CAT,
)


# ── Ablation study ──────────────────────────────────────────────────────────
@torch.no_grad()
def ablation_study(model, h5_paths: list, device: torch.device) -> np.ndarray:
    """
    For each prototype k, zero its affinity column and measure logit change.

    Uses mist._ablation_k attribute (requires the 2-line patch in mist.py).

    Returns: (K,) mean logit L2 distance across all slides.
    """
    mist = extract_mist(model)
    K = mist.K

    importance = np.zeros(K)
    n_slides = 0

    for sid, h5_path in tqdm(h5_paths, desc='Ablation'):
        features, _ = load_h5(h5_path)
        x = torch.from_numpy(features).unsqueeze(0).float().to(device)

        # Baseline logits
        mist._ablation_k = None
        base_out = model.forward_no_loss(x)
        base_logits = base_out['logits'].cpu().numpy()  # (1, C)

        # Ablate each prototype
        for k in range(K):
            mist._ablation_k = k
            abl_out = model.forward_no_loss(x)
            abl_logits = abl_out['logits'].cpu().numpy()
            importance[k] += np.linalg.norm(base_logits - abl_logits)

        n_slides += 1

    mist._ablation_k = None  # reset
    return importance / max(n_slides, 1)


# ── Slide-level profiles ───────────────────────────────────────────────────
@torch.no_grad()
def compute_slide_profiles(model, h5_paths: list,
                           device: torch.device) -> np.ndarray:
    """
    For each slide, compute mean affinity over patches → (K,) vector.

    Returns: (n_slides, K)
    """
    profiles = []
    for sid, h5_path in tqdm(h5_paths, desc='Slide profiles'):
        features, _ = load_h5(h5_path)
        out = run_inference(model, features, device)
        if 'staining_affinity' not in out:
            raise ValueError(f"No staining_affinity for {sid}")
        aff = out['staining_affinity']  # (N, K)
        profiles.append(aff.mean(axis=0))
    return np.stack(profiles)


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_ablation_bar(importance: np.ndarray, out_path: str):
    K = len(importance)
    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 4))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, K))
    ax.bar(range(K), importance, color=colors)
    ax.set_xlabel('Prototype Index')
    ax.set_ylabel('Mean Logit L2 Change')
    ax.set_title('Prototype Importance (Ablation)')
    ax.set_xticks(range(K))
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, out_path)


def plot_umap_profiles(profiles: np.ndarray, labels: np.ndarray,
                       out_path: str, label_name: str = 'class'):
    """UMAP scatter of slide-level prototype profiles."""
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean',
                        random_state=42)
    emb = reducer.fit_transform(profiles)

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap(CMAP_CAT, len(unique_labels))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(emb[mask, 0], emb[mask, 1], s=15, alpha=0.7,
                   color=cmap(i), label=str(lab), linewidths=0)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title(f'Slide Profiles by {label_name}')
    ax.legend(fontsize=8, markerscale=2)
    save_fig(fig, out_path)


def plot_profile_heatmap(profiles: np.ndarray, labels: np.ndarray,
                         out_path: str):
    """Heatmap of mean prototype profile per class."""
    unique = np.unique(labels)
    means = np.array([profiles[labels == lab].mean(axis=0) for lab in unique])
    K = means.shape[1]

    fig, ax = plt.subplots(figsize=(max(6, K * 0.5), max(3, len(unique) * 0.5)))
    im = ax.imshow(means, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'P{k}' for k in range(K)], fontsize=8)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels([str(l) for l in unique])
    ax.set_xlabel('Prototype')
    ax.set_ylabel('Class')
    ax.set_title('Mean Prototype Profile per Class')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    save_fig(fig, out_path)


def cox_regression(profiles: np.ndarray, event_times: np.ndarray,
                   censorships: np.ndarray, out_path: str) -> pd.DataFrame:
    """
    Fit univariate Cox regression for each prototype.

    Uses lifelines CoxPHFitter.
    """
    from lifelines import CoxPHFitter

    K = profiles.shape[1]
    rows = []

    for k in range(K):
        df = pd.DataFrame({
            'T': event_times,
            'E': 1 - censorships,  # lifelines: E=1 means event occurred
            f'proto_{k}': profiles[:, k],
        })
        cph = CoxPHFitter()
        cph.fit(df, duration_col='T', event_col='E')
        s = cph.summary.iloc[0]
        rows.append({
            'prototype': k,
            'coef': s['coef'],
            'hr': s['exp(coef)'],
            'p': s['p'],
            'ci_lower': s['exp(coef) lower 95%'],
            'ci_upper': s['exp(coef) upper 95%'],
        })

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)

    # Forest plot
    fig, ax = plt.subplots(figsize=(6, max(3, K * 0.35)))
    y = np.arange(K)
    ax.errorbar(result['hr'], y, xerr=[result['hr'] - result['ci_lower'],
                                        result['ci_upper'] - result['hr']],
                fmt='o', color='#4C72B0', capsize=3)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f'Proto {k}' for k in range(K)])
    ax.set_xlabel('Hazard Ratio')
    ax.set_title('Prototype Cox Regression')
    ax.grid(axis='x', alpha=0.3)
    save_fig(fig, out_path.replace('.csv', '_forest.png'))

    return result


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='MIST Prototype Importance Analysis')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data-source', type=str, nargs='+', required=True,
                        help='One or more feature directories')
    parser.add_argument('--split-csv', type=str, required=True,
                        help='CSV with slide_id and label columns')
    parser.add_argument('--config-dir', type=str, default='./configs')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'survival'])
    parser.add_argument('--label-col', type=str, default='label',
                        help='Column name for class labels')
    parser.add_argument('--time-col', type=str, default='os_survival_days',
                        help='Column for survival time (survival mode)')
    parser.add_argument('--censor-col', type=str, default='os_censorship',
                        help='Column for censorship (survival mode)')
    parser.add_argument('--max-slides', type=int, default=None,
                        help='Max slides to process (default: all)')
    parser.add_argument('--skip-ablation', action='store_true',
                        help='Skip ablation study (slow for large datasets)')
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # Load model
    model, cfg, device = load_model_from_ckpt(
        args.ckpt, args.config_dir, mode=args.mode, device_str=args.device)

    # Collect slide paths
    h5_paths = collect_h5_paths(args.data_source, args.split_csv)
    if args.max_slides:
        h5_paths = h5_paths[:args.max_slides]
    print(f"Found {len(h5_paths)} slides")

    # Read labels
    df = pd.read_csv(args.split_csv)
    id_col = 'slide_id' if 'slide_id' in df.columns else 'FILENAME'
    slide_ids_found = [sid for sid, _ in h5_paths]
    df_found = df[df[id_col].isin(slide_ids_found)].set_index(id_col)
    # Reorder to match h5_paths order
    df_found = df_found.loc[slide_ids_found]

    labels = df_found[args.label_col].values if args.label_col in df_found.columns else None

    # 1. Ablation study
    if not args.skip_ablation:
        print("\n=== Ablation Study ===")
        importance = ablation_study(model, h5_paths, device)
        plot_ablation_bar(importance, f'{args.out_dir}/ablation.png')
        pd.DataFrame({
            'prototype': range(len(importance)),
            'importance': importance,
        }).to_csv(f'{args.out_dir}/ablation.csv', index=False)
        print("Top prototypes:", np.argsort(importance)[::-1])

    # 2. Slide-level profiles
    print("\n=== Slide Profiles ===")
    profiles = compute_slide_profiles(model, h5_paths, device)
    np.save(f'{args.out_dir}/profiles.npy', profiles)

    if labels is not None:
        plot_umap_profiles(profiles, labels,
                           f'{args.out_dir}/umap_profiles.png',
                           label_name=args.label_col)
        plot_profile_heatmap(profiles, labels,
                             f'{args.out_dir}/profile_heatmap.png')

    # 3. Cox regression (survival mode)
    if args.mode == 'survival' and args.time_col in df_found.columns:
        print("\n=== Cox Regression ===")
        times = df_found[args.time_col].values.astype(float)
        censors = df_found[args.censor_col].values.astype(float)
        cox_df = cox_regression(profiles, times, censors,
                                f'{args.out_dir}/cox_regression.csv')
        print(cox_df.to_string(index=False))

    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
