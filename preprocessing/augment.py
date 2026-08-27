"""Morphology-preserving ECG augmentation, train-only (实施方案 section 8).

- Gaussian noise:  p=0.3, SNR 30-40 dB
- Global amplitude scaling: p=0.3, factor in [0.95, 1.05] (same factor for all 12 leads)
- Temporal shift: p=0.2, +-20 samples @100 Hz (all 12 leads shifted together)

Forbidden in main experiments: SMOTE / MixUp / Lead Shuffle / strong Time Warping.
"""
import numpy as np


def _gaussian_noise(x, snr_db, rng):
    sig_power = float(np.mean(x ** 2)) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    # np.sqrt() returns a float64 scalar which would upcast the float32 array;
    # cast the whole noise term to float32 before adding.
    noise = (rng.standard_normal(x.shape) * np.sqrt(noise_power)).astype(np.float32)
    return x + noise


def _amplitude_scale(x, factor):
    return x * factor


def _temporal_shift(x, shift):
    # x: [T, C]; shift in samples (positive -> right). Zero-fill the vacated edge.
    out = np.zeros_like(x)
    if shift > 0:
        out[shift:, :] = x[:-shift, :]
    elif shift < 0:
        out[:shift, :] = x[-shift:, :]
    else:
        out = x
    return out


def augment(x, rng=None, p_noise=0.3, p_scale=0.3, p_shift=0.2):
    """x: [T, C] float array. Returns augmented copy.

    ``rng`` is a ``numpy.random.Generator`` (from ``default_rng``).
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=np.float32)
    if rng.random() < p_noise:
        x = _gaussian_noise(x, float(rng.uniform(30.0, 40.0)), rng)
    if rng.random() < p_scale:
        x = _amplitude_scale(x, float(rng.uniform(0.95, 1.05)))
    if rng.random() < p_shift:
        x = _temporal_shift(x, int(rng.integers(-20, 21)))
    return x
