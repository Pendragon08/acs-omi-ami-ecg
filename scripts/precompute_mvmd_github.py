"""Precompute GitHub MVMD modes for the HMoE feasibility study.

MVMD implementation is imported verbatim from Dmocrito/mvmd.  This script is
only the project-specific cache adapter; it never reads labels and therefore
can be run before any fold is selected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "mvmd"))
from mvmd import mvmd  # noqa: E402  (GitHub: Dmocrito/mvmd)

_X = None
_K = 4


def _init_worker(x_path, shape, k):
    global _X, _K
    _X = np.memmap(x_path, mode="r", dtype="float32", shape=shape)
    _K = k


def _one(i):
    x = np.asarray(_X[i], dtype=np.float64).T  # [12, T]
    modes, _hat, _omega = mvmd(
        x, num_modes=_K, alpha=2000.0, init=1, sampling_rate=100.0,
        tolerance=1e-3, tau=0.0, verbose=False,
    )
    return i, np.asarray(modes, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", default="data/arrays/MVMD_K4_train.npy")
    args = p.parse_args()
    arr = ROOT / "data" / "arrays"
    p2 = np.load(arr / "P2_train.npy", mmap_mode="r")
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    s = scalers["P2"]
    raw = np.asarray(p2, dtype=np.float32) * (float(s["scale"]) + 1e-8) + float(s["median"])
    # A temporary raw ECG memmap prevents every worker from materialising the
    # complete P2 array in private memory.
    tmp = Path("/tmp/acs_p2_raw_mvmd.float32")
    mm = np.memmap(tmp, mode="w+", dtype="float32", shape=raw.shape)
    mm[:] = raw; mm.flush(); del mm, raw, p2
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float32",
                                    shape=(len(np.load(arr / "meta_train.npz", allow_pickle=True)["record_ids"]), args.k, 12, 1000))
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(str(tmp), tuple(out.shape[:1] + out.shape[2:]), args.k)) as ex:
        # Submit in bounded chunks so the parent does not queue all 18k arrays.
        for i, modes in ex.map(_one, range(out.shape[0]), chunksize=1):
            out[i] = modes
            done += 1
            if done % 100 == 0:
                out.flush()
                rate = done / max(1e-9, time.time() - t0)
                print(f"[MVMD] {done}/{out.shape[0]} rate={rate:.2f}/s", flush=True)
    out.flush()
    try:
        os.remove(tmp)
    except OSError:
        pass
    print(f"saved {out_path} shape={out.shape} sec={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
