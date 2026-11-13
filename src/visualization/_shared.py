"""
Shared utilities for MIST visualization tools.

Provides model loading from checkpoints, single-slide inference,
H5 feature/coord reading, and matplotlib style constants.
"""

import argparse
import os
import sys
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Plotting constants (match project conventions) ──────────────────────────
DPI = 150
CMAP_DIV = 'coolwarm'
CMAP_CAT = 'tab20'


def get_device(device_str: str = 'auto') -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


# ── CLI helpers ─────────────────────────────────────────────────────────────
def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config-dir', type=str, default='./configs',
                        help='Directory containing model config folders')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'survival'],
                        help='Model mode (affects n_classes reconstruction)')
    return parser


# ── Model loading ───────────────────────────────────────────────────────────
def load_model_from_ckpt(ckpt_path: str, config_dir: str = './configs',
                         mode: str = 'classification',
                         device_str: str = 'auto'):
    """
    Reconstruct a trained model from its checkpoint.

    Checkpoint format (from utils.utils.save_checkpoint):
        {'model': state_dict, 'config': vars(args), 'epoch': int, 'score': float}

    Returns: (model, ckpt_config_dict, device)
    """
    # Ensure mil_models is importable
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from mil_models.model_factory import create_downstream_model

    device = get_device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device)

    # The checkpoint may store state_dict directly (no early stopping) or as a dict
    if isinstance(ckpt, dict) and 'config' in ckpt:
        cfg = ckpt['config']
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # Plain state_dict — need config_dir to figure out model
        raise ValueError(
            "Checkpoint has no 'config' key. Was it saved with "
            "torch.save(model.state_dict(), ...)? Use a checkpoint from "
            "early-stopping (s_checkpoint.pth) that includes config.")
    else:
        raise ValueError(f"Unrecognised checkpoint format: {type(ckpt)}")

    # Rebuild argparse Namespace from saved config dict
    args = argparse.Namespace(**cfg)

    # Override mode if the caller requests a different one
    # (the saved config remembers the original training mode)
    model = create_downstream_model(args, mode=mode, config_dir=config_dir)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg, device


def extract_mist(model):
    """Get the MolecularStainingTransform submodule from any supported model."""
    m = model.module if hasattr(model, 'module') else model
    if not hasattr(m, 'mist'):
        raise ValueError("Model does not have a MIST module (use_mist=False in config?)")
    return m.mist


# ── Data loading ────────────────────────────────────────────────────────────
def load_h5(h5_path: str):
    """
    Load features and coordinates from an H5 file.

    Returns: (features: np.ndarray (N, D), coords: np.ndarray (N, 2))
    """
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        coords = f['coords'][:] if 'coords' in f else np.empty((0, 2))
    # Squeeze leading singleton dim if present
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)
    return features, coords


def collect_h5_paths(data_sources, split_csv, use_h5=True):
    """
    Collect h5 file paths for all slides in a split CSV.

    Args:
        data_sources: list of dataroot directories
        split_csv: path to CSV file with 'slide_id' column
        use_h5: if True look for .h5, else .pt

    Returns: list of (slide_id, h5_path) tuples
    """
    import pandas as pd
    df = pd.read_csv(split_csv)
    id_col = 'slide_id' if 'slide_id' in df.columns else 'FILENAME'
    slide_ids = df[id_col].tolist()

    ext = '.h5' if use_h5 else '.pt'
    results = []
    for sid in slide_ids:
        found = False
        for ds in data_sources:
            # Walk common directory patterns
            for sub in ['feats_h5', 'feats_pt', '']:
                candidate = os.path.join(ds, sub, sid + ext)
                if os.path.isfile(candidate):
                    results.append((sid, candidate))
                    found = True
                    break
                # Also try without extension if already present
                candidate2 = os.path.join(ds, sub, sid)
                if os.path.isfile(candidate2):
                    results.append((sid, candidate2))
                    found = True
                    break
            if found:
                break
        if not found:
            # Try globbing recursively
            import glob
            matches = glob.glob(os.path.join(ds, '**', sid + ext), recursive=True)
            if matches:
                results.append((sid, matches[0]))
    return results


# ── Inference ───────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, features: np.ndarray, device: torch.device) -> dict:
    """
    Run forward_no_loss on a single slide (B=1).

    Returns dict with numpy arrays (batch dim squeezed).
    """
    x = torch.from_numpy(features).unsqueeze(0).float().to(device)
    out = model.forward_no_loss(x)
    result = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu().numpy().squeeze(0)
        else:
            result[k] = v
    return result


# ── Plotting helpers ────────────────────────────────────────────────────────
def save_fig(fig, path: str, close: bool = True):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    if close:
        plt.close(fig)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
