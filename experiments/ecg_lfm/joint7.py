"""Authoritative Joint-7 label mapping and probability marginalization."""
from __future__ import annotations

import numpy as np
import torch


ACS_CLASSES = ("OTHER", "STEMI", "NSTEMI", "UA")
JOINT_CLASSES = (
    "OTHER",
    "STEMI+nOMI",
    "STEMI+OMI",
    "NSTEMI+nOMI",
    "NSTEMI+OMI",
    "UA+nOMI",
    "UA+OMI",
)
RAW_COMBINATIONS = tuple(
    f"{acs}+{'OMI' if omi else 'nOMI'}" for acs in ACS_CLASSES for omi in (0, 1)
)


def encode_source_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map [STEMI,NSTEMI,UA,OMI] to ACS, OMI, Joint-7, exception."""
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError(f"source labels must have shape [N,4], got {y.shape}")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("source labels must be finite binary values")
    subtype_sum = y[:, :3].sum(axis=1)
    if np.any(subtype_sum > 1):
        raise ValueError("STEMI/NSTEMI/UA overlap violates categorical ACS")
    acs = np.where(subtype_sum == 0, 0, y[:, :3].argmax(axis=1) + 1).astype(np.int64)
    omi = y[:, 3].astype(np.int64)
    exception = (acs == 0) & (omi == 1)
    joint = np.zeros(len(y), dtype=np.int64)
    joint[(acs == 1) & (omi == 0)] = 1
    joint[(acs == 1) & (omi == 1)] = 2
    joint[(acs == 2) & (omi == 0)] = 3
    joint[(acs == 2) & (omi == 1)] = 4
    joint[(acs == 3) & (omi == 0)] = 5
    joint[(acs == 3) & (omi == 1)] = 6
    # OTHER+OMI is deliberately represented by OTHER for the categorical loss;
    # its original OMI label and exception flag remain independent ground truth.
    joint[exception] = 0
    return acs, omi, joint, exception


def one_hot_joint7(joint: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.int64)
    if joint.ndim != 1 or np.any((joint < 0) | (joint >= 7)):
        raise ValueError("joint target must contain class indices 0..6")
    return np.eye(7, dtype=np.float32)[joint]


def marginalize_joint_probabilities(q):
    """Return ACS, OMI, NSTE and conditional probabilities from Joint-7 q."""
    if q.shape[-1] != 7:
        raise ValueError(f"expected seven joint probabilities, got {q.shape}")
    components = (q[..., 0], q[..., 1] + q[..., 2], q[..., 3] + q[..., 4], q[..., 5] + q[..., 6])
    acs = torch.stack(components, dim=-1) if torch.is_tensor(q) else np.stack(components, axis=-1)
    p_omi = q[..., 2] + q[..., 4] + q[..., 6]
    p_nste = q[..., 3] + q[..., 4] + q[..., 5] + q[..., 6]
    p_nste_omi = q[..., 4] + q[..., 6]
    eps = torch.finfo(q.dtype).eps if torch.is_tensor(q) else np.finfo(q.dtype).eps
    p_omi_given_nste = p_nste_omi / (p_nste + eps)
    s_nstemi = q[..., 3] + q[..., 4]
    s_ua = q[..., 5] + q[..., 6]
    p_nstemi_given_nste = s_nstemi / (s_nstemi + s_ua + eps)
    return {
        "acs": acs,
        "omi": p_omi,
        "nste": p_nste,
        "nste_omi": p_nste_omi,
        "omi_given_nste": p_omi_given_nste,
        "nstemi_given_nste": p_nstemi_given_nste,
    }
