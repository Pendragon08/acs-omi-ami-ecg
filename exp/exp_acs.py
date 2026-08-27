"""Multi-label ACS trainer (ASL / BCE, sigmoid, patient-level split).

Unified training loop used by all 6 experiments:

    [B,1000,12] ECG -> model -> [B,C] logits -> sigmoid -> P(OMI[,AMI])
                                   \\-> Asymmetric Loss (or BCE)

Per seed it (1) trains with early-stopping on validation macro-AUROC,
(2) selects per-label thresholds on validation by maximizing F1,
(3) reports the full metric table on the held-out validation set using those
thresholds (the official test.csv is used only for final prediction), and
(4) dumps a JSON result + per-epoch CSV + best checkpoint.

Run standalone (smoke test):
    python -m exp.exp_acs --model TimesNet --variant P3 --seed 2026 --epochs 3
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent

from nets import build_model                                  # noqa: E402
from losses.asl import AsymmetricLoss                         # noqa: E402
from losses.focal import FocalLoss                            # noqa: E402
from acs_utils.seed import set_seed                           # noqa: E402
from acs_utils.metrics import compute_metrics, flatten_metrics, LABELS  # noqa: E402
from acs_utils.threshold import select_thresholds             # noqa: E402
from data_provider.acs_dataset import ACSDataset              # noqa: E402


# Task presets: the historical OMI/AMI tasks plus a three-label clinical task.
# --task omi        -> [B,1] output, label OMI
# --task ami        -> [B,1] output, label AMI
# --task multilabel -> [B,2] output, labels OMI+AMI
# --task stemi_nstemi_ua -> [B,3] output, labels STEMI/NSTEMI/UA
TASK_MAP = {
    "omi": (1, [0], ["OMI"]),
    "ami": (1, [1], ["AMI"]),
    "multilabel": (2, [0, 1], ["OMI", "AMI"]),
    "stemi_nstemi_ua": (3, [2, 3, 4], ["STEMI", "NSTEMI", "UA"]),
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _load_split():
    with open(ROOT / "data" / "splits" / "patient_split.json") as f:
        return json.load(f)


def _load_meta():
    """Load array metadata and align the CSV clinical labels by record id.

    The historical ``meta_train.npz`` contains only OMI/AMI columns.  The
    source CSV contains the additional STEMI/NSTEMI/UA labels, so append those
    columns in the same record order instead of assuming CSV row order equals
    the mmap array order.  The original OMI/AMI columns remain unchanged.
    """
    raw = np.load(ROOT / "data" / "arrays" / "meta_train.npz", allow_pickle=True)
    record_ids = np.asarray(raw["record_ids"]).astype(str)
    labels = np.asarray(raw["labels"], dtype=np.int64)
    csv_labels = {}
    with (ROOT / "data" / "CSV" / "train.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rid = row["ecg_row_record"].split(".")[0]
            csv_labels[rid] = [int(row[k]) for k in ("STEMI", "NSTEMI", "UA")]
    extra = np.asarray([csv_labels[rid] for rid in record_ids], dtype=np.int64)
    return {
        "record_ids": record_ids,
        "patient_ids": np.asarray(raw["patient_ids"]),
        "labels": np.column_stack([labels, extra]),
    }


def make_loaders(variant, batch_size, seed, num_workers=4, label_cols=(0, 1),
                 split=None, meta=None, use_fusion=False):
    """Build train/val DataLoaders for a preprocessing variant.

    P0/P1/P2 use the prebuilt array; P3 = P2 array + online augmentation.
    ``use_fusion`` additionally loads the median-beat and clinical-feature
    arrays, so each batch becomes (x, med, clin, y).
    """
    if split is None:
        split = _load_split()
    if meta is None:
        meta = _load_meta()

    array_key = "P2" if variant == "P3" else variant
    arr_path = ROOT / "data" / "arrays" / f"{array_key}_train.npy"
    labels = meta["labels"][:, list(label_cols)].astype(np.float32)

    med_path = clin_path = None
    if use_fusion:
        med_path = ROOT / "data" / "arrays" / "med_P2_train.npy"
        clin_path = ROOT / "data" / "arrays" / "clin_train.npy"

    train_ds = ACSDataset(arr_path, labels=labels,
                          augment=(variant == "P3"), rng_seed=seed,
                          med_path=med_path, clin_path=clin_path)
    val_ds = ACSDataset(arr_path, labels=labels, augment=False,
                        med_path=med_path, clin_path=clin_path)

    train_ds = torch.utils.data.Subset(train_ds, split["train"])
    val_ds = torch.utils.data.Subset(val_ds, split["val"])

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              generator=g, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Criterion
# --------------------------------------------------------------------------- #
def make_criterion(loss_type, pos_weight=None, class_weight=None, device="cpu",
                   focal_gamma=2.0, focal_alpha=None):
    """Build the loss.  All losses operate on [B,1] raw logits (single-label).

    loss_type:
      asl   -> AsymmetricLoss (gamma_pos=0, gamma_neg=4, clip=0.05), sum / B
      bce   -> plain BCEWithLogitsLoss (no weighting)
      wbce  -> BCEWithLogitsLoss(pos_weight=pw)           (data-driven ratio)
      wce   -> binary form of CrossEntropyLoss(weight=[w_neg, w_pos]).
               For a 2-class softmax, CE(weight=[w_neg, w_pos]) is exactly
               equivalent to BCEWithLogitsLoss(pos_weight=w_pos/w_neg) on a
               single logit, so we implement it that way for numerical stability.
      focal -> FocalLoss(gamma=focal_gamma, alpha=focal_alpha), mean / B.
    """
    if loss_type == "asl":
        asl = AsymmetricLoss()
        def _asl(out, y):
            return asl(out, y) / out.size(0)
        return _asl
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    if loss_type == "wbce":
        pw = float(pos_weight)
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], dtype=torch.float32, device=device))
    if loss_type == "wce":
        w_neg, w_pos = float(class_weight[0]), float(class_weight[1])
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([w_pos / w_neg], dtype=torch.float32,
                                    device=device))
    raise ValueError(f"unknown loss: {loss_type}")


# --------------------------------------------------------------------------- #
# Core loop
# --------------------------------------------------------------------------- #
def predict(model, loader, device, use_fusion=False):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if use_fusion:
                med = batch[1].to(device)
                clin = batch[2].to(device)
                y = batch[3]
                out = model(x, med=med, clin=clin)
            else:
                y = batch[1]
                out = model(x)
            ps.append(torch.sigmoid(out).cpu().numpy())
            ys.append(y.cpu().numpy())
    return np.concatenate(ys, 0), np.concatenate(ps, 0)


def train_and_evaluate(args):
    set_seed(args.seed)
    device = args.device
    label_names = tuple(args.label_names)
    use_fusion = getattr(args, "use_fusion", False)

    # ---- data ---------------------------------------------------------------
    train_loader, val_loader = make_loaders(
        args.variant, args.batch_size, args.seed,
        num_workers=getattr(args, "num_workers", 4),
        label_cols=tuple(args.label_cols),
        use_fusion=use_fusion,
    )

    # ---- model / optimizer / criterion -------------------------------------
    # Capacity overrides (model-size sweep): only pass the hyperparams that were
    # explicitly set; everything else falls back to MODEL_DEFAULTS.
    overrides = {k: getattr(args, k) for k in ("d_model", "d_ff", "e_layers", "n_heads")
                 if getattr(args, k, None) is not None}
    model = build_model(args.model, num_class=args.num_class, device=device, **overrides)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = make_criterion(
        args.loss,
        pos_weight=getattr(args, "pos_weight", None),
        class_weight=getattr(args, "class_weight", None),
        device=device,
        focal_gamma=getattr(args, "focal_gamma", 2.0),
        focal_alpha=getattr(args, "focal_alpha", None),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- train --------------------------------------------------------------
    best_val_auroc = -1.0
    best_state = None
    best_thresholds = None
    best_epoch = -1
    patience_counter = 0
    history = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batch = 0
        for batch in train_loader:
            x = batch[0].to(device)
            if use_fusion:
                med = batch[1].to(device)
                clin = batch[2].to(device)
                y = batch[3].to(device)
            else:
                y = batch[1].to(device)
            optimizer.zero_grad()
            out = model(x) if not use_fusion else model(x, med=med, clin=clin)
            loss = criterion(out, y)
            loss.backward()
            if getattr(args, "clip", 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            running += float(loss.item())
            n_batch += 1

        train_loss = running / max(n_batch, 1)

        # validation
        y_val, p_val = predict(model, val_loader, device, use_fusion=use_fusion)
        val_metrics = compute_metrics(y_val, p_val, thresholds=None,
                                      label_names=label_names)
        val_auroc = val_metrics["macro_auroc"]
        thr = select_thresholds(y_val, p_val)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_auroc": val_auroc,
                        "val_auprc": val_metrics["macro_auprc"],
                        "val_f1": val_metrics["macro_f1"]})

        print(f"[seed {args.seed}] epoch {epoch}/{args.epochs} "
              f"loss={train_loss:.4f} val_macro_auroc={val_auroc:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thresholds = thr
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[seed {args.seed}] early stop at epoch {epoch} "
                      f"(best={best_epoch})", flush=True)
                break

    train_time = time.time() - t0

    # ---- evaluate best model on held-out validation set --------------------
    model.load_state_dict(best_state)
    model.to(device)
    val_metrics_at_best = compute_metrics(
        *predict(model, val_loader, device, use_fusion=use_fusion),
        thresholds=best_thresholds, label_names=label_names)

    peak_mem = (torch.cuda.max_memory_allocated(device) / 2**20
                if device.startswith("cuda") else 0.0)

    return {
        "model": args.model,
        "seed": args.seed,
        "loss": args.loss,
        "variant": args.variant,
        "use_fusion": use_fusion,
        "pos_weight": getattr(args, "pos_weight", None),
        "class_weight": (list(args.class_weight)
                         if getattr(args, "class_weight", None) else None),
        "lr": args.lr,
        "clip": getattr(args, "clip", 0.0),
        "num_class": args.num_class,
        "label_names": list(label_names),
        "best_epoch": best_epoch,
        "n_params": int(n_params),
        "train_time_sec": round(train_time, 2),
        "peak_mem_mb": round(peak_mem, 2),
        "thresholds": best_thresholds.tolist(),
        "val": flatten_metrics(val_metrics_at_best),
        "history": history,
        "_state": best_state,
    }


def run_exp(args, out_dir=None):
    """Train one (model, seed) and dump result JSON. Returns the result dict."""
    out_dir = Path(out_dir) if out_dir else ROOT / "results" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Single-label runs must carry the label name in the tag, otherwise the
    # OMI and AMI runs of the same model would write to the same JSON file.
    suffix = getattr(args, "tag_suffix", "") or ""
    if suffix:
        suffix = f"_{suffix}"
    if args.num_class == 1 and args.label_names:
        tag = f"{args.model}_{args.label_names[0]}_{args.loss}_{args.variant}_seed{args.seed}{suffix}"
    else:
        tag = f"{args.model}_{args.loss}_{args.variant}_seed{args.seed}{suffix}"

    result = train_and_evaluate(args)

    # best checkpoint (state dict kept out of the JSON)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = result.pop("_state", None)
    if state is not None:
        torch.save(state, ckpt_dir / f"{tag}.pt")

    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2),
                                         encoding="utf-8")
    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TimesNet")
    p.add_argument("--task", default=None,
                   choices=["omi", "ami", "multilabel", "stemi_nstemi_ua"],
                   help="task preset: omi/ami = single-label [B,1]; "
                        "multilabel = [B,2] OMI+AMI (disabled by default)")
    p.add_argument("--num_class", type=int, default=2)
    p.add_argument("--loss", default="asl", choices=["asl", "bce", "wce", "wbce", "focal"])
    p.add_argument("--pos_weight", type=float, default=None,
                   help="positive-class weight for --loss wbce")
    p.add_argument("--class_weight", type=float, nargs=2, default=None,
                   metavar=("W_NEG", "W_POS"),
                   help="class weights [w_neg, w_pos] for --loss wce")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="focusing parameter for --loss focal")
    p.add_argument("--focal_alpha", type=float, default=None,
                   help="positive-class weight for --loss focal (None = no balancing)")
    p.add_argument("--use_fusion", action="store_true",
                   help="multimodal input: raw waveform + median beat + clinical features")
    p.add_argument("--variant", default="P3", choices=["P0", "P1", "P2", "P3"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0,
                   help="gradient norm clipping (<=0 disables); guards against "
                        "ASL collapse under extreme OMI imbalance")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--label_cols", type=int, nargs="+", default=[0, 1])
    p.add_argument("--label_names", nargs="+", default=list(LABELS))
    p.add_argument("--exp_name", default="smoke")
    p.add_argument("--tag_suffix", default="",
                   help="append to result tag (e.g. lr5e-5_bs256 for hyperparam tuning)")
    p.add_argument("--device", default=None)
    # ---- capacity sweep (override MODEL_DEFAULTS; None = use default) -------
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--e_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.task:
        args.num_class, args.label_cols, args.label_names = TASK_MAP[args.task]
    if len(args.label_names) != args.num_class:
        args.label_names = args.label_names[:args.num_class]
    result = run_exp(args)
    print(json.dumps({"best_epoch": result["best_epoch"],
                      "val": result["val"]}, indent=2))


if __name__ == "__main__":
    main()
