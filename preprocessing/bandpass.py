"""0.5-40 Hz 4th-order zero-phase Butterworth bandpass (fs = 100 Hz).

实施方案 section 6: scipy.signal.butter(output='sos') + sosfiltfilt, applied
per lead along the time axis.
"""
from scipy.signal import butter, sosfiltfilt


def build_bandpass_sos(fs=100.0, low=0.5, high=40.0, order=4):
    return butter(order, [low, high], btype="band", fs=fs, output="sos")


def bandpass_filter(x, fs=100.0, low=0.5, high=40.0, order=4):
    """Filter along the time axis (axis=1 for [N,T,C]; axis=-1 for [T,C])."""
    sos = build_bandpass_sos(fs, low, high, order)
    axis = 1 if x.ndim == 3 else -1
    return sosfiltfilt(sos, x, axis=axis)
