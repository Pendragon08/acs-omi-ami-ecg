"""Experiment 2 — 7-model multi-label comparison (方案 section 20).

3 modern (TimesNet / Informer / Nonstationary_Transformer)
+ 4 traditional (LSTM / ResNet1D_Wang / InceptionTime / ViT), all [B,2]
OMI+AMI, ASL, P3, 5 seeds.
"""
# LEGACY ONLY: this historical OMI+AMI runner depends on the removed
# experiments.common module and is not valid for the corrected ACS categorical
# + OMI binary task. Use experiments.acs_joint_model_comparison.py instead.
from experiments.common import aggregate, run_matrix

MODELS = [
    "TimesNet", "Informer", "Nonstationary_Transformer",
    "LSTM", "ResNet1D_Wang", "InceptionTime", "ViT",
]


def combos():
    return [dict(model=m) for m in MODELS]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--no_aggregate", action="store_true")
    a = p.parse_args()

    models = a.models or MODELS
    seeds = tuple(a.seeds) if a.seeds else (2026, 2027, 2028, 2029, 2030)
    run_matrix("exp2", [dict(model=m) for m in models], seeds=seeds, epochs=a.epochs)
    if not a.no_aggregate:
        aggregate("exp2")
