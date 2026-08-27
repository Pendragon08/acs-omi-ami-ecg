"""Global Robust Scaling (实施方案 section 7).

A single median and a single scale (1.4826 * MAD) computed over the *entire*
training set (all records and all 12 leads flattened together).  All leads share
the same median + scale, so relative inter-lead amplitudes are preserved.
Validation / test only apply the train scaler.
"""
import numpy as np


def fit_robust_scaler(X_train):
    """X_train: [N, T, C] -> {'median': float, 'scale': float}."""
    flat = np.asarray(X_train, dtype=np.float64).reshape(-1)
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    scale = 1.4826 * mad
    return {"median": median, "scale": scale}


def apply_robust_scaler(X, scaler, eps=1e-8):
    """Apply a pre-fit scaler to X [N,T,C] (or [T,C])."""
    X = np.asarray(X, dtype=np.float64)
    return ((X - scaler["median"]) / (scaler["scale"] + eps)).astype(np.float32)
