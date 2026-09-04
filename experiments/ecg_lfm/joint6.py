"""Authoritative observed Joint-6 mapping and probability marginalization."""
from __future__ import annotations

import numpy as np
import torch


ACS_CLASSES = ("OTHER", "STEMI", "NSTEMI", "UA")
JOINT6_CLASSES = (
    "OTHER",
    "STEMI+nOMI",
    "STEMI+OMI",
    "NSTEMI+nOMI",
    "NSTEMI+OMI",
    "UA",
)


def encode_source_labels(
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map [STEMI,NSTEMI,UA,OMI] to ACS, OMI, Joint-6 and exception.

    The sole OTHER+OMI record maps to OTHER for categorical training while its
    raw OMI target remains positive for primary OMI evaluation. UA+OMI is a
    hard error because that unsupported state is not present in Joint-6.
    """
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
    if np.any((acs == 3) & (omi == 1)):
        raise ValueError("UA+OMI is unsupported by the observed Joint-6 label space")
    exception = (acs == 0) & (omi == 1)
    joint = np.zeros(len(y), dtype=np.int64)
    joint[(acs == 1) & (omi == 0)] = 1
    joint[(acs == 1) & (omi == 1)] = 2
    joint[(acs == 2) & (omi == 0)] = 3
    joint[(acs == 2) & (omi == 1)] = 4
    joint[acs == 3] = 5
    joint[exception] = 0
    return acs, omi, joint, exception


def one_hot_joint6(joint: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.int64)
    if joint.ndim != 1 or np.any((joint < 0) | (joint >= 6)):
        raise ValueError("joint target must contain class indices 0..5")
    return np.eye(6, dtype=np.float32)[joint]


def marginalize_joint6_probabilities(q):
    """Recover ACS, OMI and NSTE probabilities from Joint-6 softmax."""
    if q.shape[-1] != 6:
        raise ValueError(f"expected six joint probabilities, got {q.shape}")
    components = (
        q[..., 0],
        q[..., 1] + q[..., 2],
        q[..., 3] + q[..., 4],
        q[..., 5],
    )
    acs = torch.stack(components, dim=-1) if torch.is_tensor(q) else np.stack(components, axis=-1)
    p_omi = q[..., 2] + q[..., 4]
    p_nste = q[..., 3] + q[..., 4] + q[..., 5]
    p_nste_omi = q[..., 4]
    eps = torch.finfo(q.dtype).eps if torch.is_tensor(q) else np.finfo(q.dtype).eps
    p_omi_given_nste = p_nste_omi / (p_nste + eps)
    p_nstemi_given_nste = (q[..., 3] + q[..., 4]) / (p_nste + eps)
    return {
        "acs": acs,
        "omi": p_omi,
        "nste": p_nste,
        "nste_omi": p_nste_omi,
        "omi_given_nste": p_omi_given_nste,
        "nstemi_given_nste": p_nstemi_given_nste,
    }
