"""Generate external four-label predictions for the traditional comparison models.

The models and preprocessing are the registered four-label comparison run.  For
each model this script loads the five frozen fold checkpoints, applies the
corresponding fold-fitted robust scaler to the official external P2 array,
averages record-level probabilities, and applies the model's pooled-OOF
Youden thresholds.  It never reads external labels.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import hmoe_kan_github as h  # noqa: E402
from experiments.hmoe_kan_e4_4label_multilabel import LABELS  # noqa: E402
from experiments.model_comparison_4label import build_baseline_model  # noqa: E402


def stem(value: str) -> str:
    return Path(str(value)).stem


def recover_external_raw() -> tuple[np.ndarray, np.ndarray]:
    arr = ROOT / "data" / "arrays"
    p2 = np.load(arr / "P2_test.npy", mmap_mode="r")
    scalers = json.loads((arr / "scalers.json").read_text(encoding="utf-8"))
    p2_scaler = scalers["P2"]
    raw = p2.astype(np.float32) * (float(p2_scaler["scale"]) + 1e-8) + float(p2_scaler["median"])
    meta = np.load(arr / "meta_test.npz", allow_pickle=True)
    return raw, np.asarray([stem(value) for value in meta["record_ids"]], dtype=object)


def test_order(meta_ids: np.ndarray) -> tuple[list[str], np.ndarray]:
    csv_path = ROOT / "data" / "CSV" / "test.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        ids_out = [(row.get("ecg_row_record") or "").strip() for row in csv.DictReader(handle)]
    meta_index = {str(value): index for index, value in enumerate(meta_ids)}
    requested = [stem(value) for value in ids_out]
    if len(requested) != len(meta_ids) or set(requested) != set(meta_index):
        raise ValueError("test.csv identifiers do not exactly match meta_test identifiers")
    return ids_out, np.asarray([meta_index[value] for value in requested], dtype=np.int64)


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size], dtype=np.float32)).to(
                device, non_blocking=True
            )
            output = model(xb)
            if isinstance(output, dict):
                output = output["logits"]
            if output.ndim != 2 or output.shape[1] != len(LABELS):
                raise RuntimeError("expected [batch, 4] logits, got %r" % (tuple(output.shape),))
            values.append(torch.sigmoid(output).cpu().numpy())
    return np.concatenate(values, axis=0)


def run_model(model_name: str, result_root: Path, x_raw: np.ndarray, meta_ids: np.ndarray,
              ids_out: list[str], order: np.ndarray, device: torch.device, batch_size: int) -> None:
    model_root = result_root / model_name
    summary = json.loads((model_root / "summary.json").read_text(encoding="utf-8"))
    thresholds = np.asarray(summary["thresholds"]["youden_j"], dtype=np.float32)
    if thresholds.shape != (len(LABELS),):
        raise ValueError("invalid threshold vector for %s: %r" % (model_name, thresholds.tolist()))

    fold_probabilities = []
    for fold_idx in range(5):
        fold_root = model_root / ("fold%d" % fold_idx)
        scaler = json.loads((fold_root / "scaler.json").read_text(encoding="utf-8"))
        x_test = h.apply_robust_scaler(x_raw, scaler)
        model = build_baseline_model(model_name).to(device)
        state = torch.load(fold_root / "best_model.pt", map_location=device)
        model.load_state_dict(state)
        fold_probabilities.append(predict(model, x_test, device, batch_size))
        del model, state, x_test
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Fold probabilities are in meta_test order; restore the exact test.csv order.
    probabilities = np.mean(np.stack(fold_probabilities, axis=0), axis=0)[order]
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)
    internal = model_root / "external_test_predictions.csv"
    with internal.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ecg_row_record"] + ["p_" + name for name in LABELS] + ["pred_youden_" + name for name in LABELS])
        for record_id, probability, prediction in zip(ids_out, probabilities, predictions):
            writer.writerow([record_id, *probability.tolist(), *prediction.astype(int).tolist()])

    submission = model_root / ("submission_4label_%s.csv" % model_name)
    with submission.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ecg_row_record", *LABELS])
        for record_id, prediction in zip(ids_out, predictions):
            writer.writerow([record_id, *prediction.astype(int).tolist()])

    counts = {name: int(predictions[:, index].sum()) for index, name in enumerate(LABELS)}
    (model_root / "external_prediction_provenance.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "n_external_records": int(len(ids_out)),
                "fold_count": 5,
                "aggregation": "arithmetic mean of five fold record-level probabilities",
                "threshold_method": "pooled OOF validation-only Youden J",
                "thresholds": {name: float(thresholds[index]) for index, name in enumerate(LABELS)},
                "predicted_positive_counts": counts,
                "external_labels_used": False,
                "submission": str(submission),
                "internal_predictions": str(internal),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"model": model_name, "submission": str(submission), "n_positive": counts}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="InceptionTime,ResNet1D_Wang,LSTM,Transformer1D,ViT")
    parser.add_argument("--result_root", default="results/model_comparison_4label")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result_root = ROOT / args.result_root
    x_raw, meta_ids = recover_external_raw()
    ids_out, order = test_order(meta_ids)
    for model_name in [value.strip() for value in args.models.split(",") if value.strip()]:
        run_model(model_name, result_root, x_raw, meta_ids, ids_out, order, device, args.batch_size)


if __name__ == "__main__":
    main()
