"""Experiment 3 — loss comparison: BCE vs ASL (方案 section 22).

TimesNet, [B,2] OMI+AMI, P3, 5 seeds per loss.
"""
from experiments.common import aggregate, run_matrix


def combos():
    return [
        dict(loss="bce"),
        dict(loss="asl"),
    ]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--no_aggregate", action="store_true")
    a = p.parse_args()

    seeds = tuple(a.seeds) if a.seeds else (2026, 2027, 2028, 2029, 2030)
    run_matrix("exp3", combos(), seeds=seeds, epochs=a.epochs)
    if not a.no_aggregate:
        aggregate("exp3")
