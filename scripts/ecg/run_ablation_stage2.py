"""Run five-seed confirmation for validation-selected losses and features."""
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/loss_feature_ablation"
LOG_DIR = ROOT / "logs/loss_feature_ablation"
SEEDS = [42, 2024, 2025, 2026, 3407]
PYTHON = "/data/code/ouyanghao/miniconda3/envs/oyh/bin/python"


def run(tag, args, expected_json):
    if expected_json.exists():
        print(f"SKIP existing {expected_json.name}", flush=True)
        return
    command = [PYTHON, "-m", "exp.exp_acs_ablation", *args]
    print("RUN", " ".join(command), flush=True)
    with open(LOG_DIR / f"{tag}.log", "w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, env=os.environ.copy())
    print(f"DONE {tag}", flush=True)


def common(study, task, seed):
    if task == "omi":
        lr, batch = "3e-4", "256"
    else:
        lr, batch = "1e-4", "128"
    return [
        "--study", study, "--task", task, "--seed", str(seed), "--lr", lr,
        "--batch_size", batch, "--epochs", "100", "--patience", "15",
        "--clip", "1.0", "--num_workers", "4", "--device", "cuda",
        "--exp_name", "loss_feature_ablation",
    ]


def focal_args(selected):
    source_path = ROOT / selected["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))["loss_config"]
    args = ["--loss", "focal", "--gamma", str(selected["gamma"])]
    if source.get("alpha_source") == "training_set_negative_fraction":
        args += ["--focal_alpha_mode", "ratio"]
        detail = f"g{selected['gamma']:g}_aratio"
    else:
        args += ["--focal_alpha", str(selected["alpha"])]
        alpha = str(selected["alpha"]).replace(".", "p")
        detail = f"g{selected['gamma']:g}_a{alpha}"
    return args, detail


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    selection_path = RESULT_DIR / "stage1_selection.json"
    if not selection_path.exists():
        raise SystemExit("stage1_selection.json missing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    focal_extra, focal_detail = focal_args(selection["best_focal"])
    cb = selection["best_cb_focal"]
    loss_specs = [
        ("bce", ["--loss", "bce"], "bce_plain"),
        ("wbce", ["--loss", "wbce"], "wbce_auto"),
        ("focal", focal_extra, f"focal_{focal_detail}"),
        (
            "cb_focal",
            ["--loss", "cb_focal", "--beta", str(cb["beta"]), "--gamma", str(cb["gamma"])],
            f"cb_focal_b{cb['beta']:g}_g{cb['gamma']:g}",
        ),
    ]
    for loss_name, extra, detail in loss_specs:
        for seed in SEEDS:
            expected = RESULT_DIR / f"loss_{detail}_omi_seed{seed}.json"
            run(f"confirm_loss_{loss_name}_omi_seed{seed}", common("loss", "omi", seed) + extra, expected)

    for task in ("omi", "ami"):
        best_mode = selection["best_added_feature"][task]["feature_mode"]
        for mode in ("raw", best_mode):
            for seed in SEEDS:
                expected = RESULT_DIR / f"feature_{mode}_{task}_seed{seed}.json"
                run(
                    f"confirm_feature_{mode}_{task}_seed{seed}",
                    common("feature", task, seed) + ["--loss", "bce", "--feature_mode", mode],
                    expected,
                )
    print("STAGE2_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
