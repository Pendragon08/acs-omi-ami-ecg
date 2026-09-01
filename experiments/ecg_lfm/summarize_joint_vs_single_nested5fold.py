"""Aggregate completed nested Joint/ACS-only/OMI-only outer folds safely."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from experiments.acs_joint_task import ACS_CLASSES, acs_metrics, nstemi_vs_ua_metrics


ROOT = Path(__file__).resolve().parents[2]
ACS_KEYS = (
    "macro_auroc", "macro_auprc", "macro_f1", "balanced_accuracy", "accuracy",
    "nstemi_auprc", "nstemi_f1", "nstemi_sensitivity", "nstemi_vs_ua_auprc",
)
OMI_KEYS = (
    "auroc", "auprc", "f1", "sensitivity", "specificity", "ppv", "npv",
    "mcc", "accuracy", "threshold", "nste_auprc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path, fold: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(int(row["fold"]) != fold or row["subset"] != "outer_test" for row in rows):
        raise ValueError(f"invalid outer-test prediction metadata: {path}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate record_id within {path}")
    return rows


def acs_fold(metrics: dict) -> dict[str, float]:
    outer = metrics["outer_test"]
    acs = outer["acs"]
    nstemi = acs["per_class"]["NSTEMI"]
    return {
        "macro_auroc": float(acs["macro_auroc_ovr"]),
        "macro_auprc": float(acs["macro_auprc_ovr"]),
        "macro_f1": float(acs["macro_f1"]),
        "balanced_accuracy": float(acs["balanced_accuracy"]),
        "accuracy": float(acs["accuracy"]),
        "nstemi_auprc": float(nstemi["auprc"]),
        "nstemi_f1": float(nstemi["f1"]),
        "nstemi_sensitivity": float(nstemi["sensitivity"]),
        "nstemi_vs_ua_auprc": float(outer["nstemi_vs_ua"]["auprc"]),
    }


def omi_fold(metrics: dict) -> dict[str, float]:
    outer = metrics["outer_test"]
    omi = outer["omi"]["youden_j"]
    return {
        "auroc": float(omi["auroc"]), "auprc": float(omi["auprc"]),
        "f1": float(omi["f1"]), "sensitivity": float(omi["sensitivity"]),
        "specificity": float(omi["specificity"]), "ppv": float(omi["ppv"]),
        "npv": float(omi["npv"]), "mcc": float(omi["mcc"]),
        "accuracy": float(omi["accuracy"]), "threshold": float(omi["threshold"]),
        "nste_auprc": float(outer["omi_among_nste_acs"]["metrics"]["auprc"]),
    }


def mean_std(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    output = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        output[key] = {
            "mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "values": [float(value) for value in values],
        }
    return output


def aligned(left: list[dict[str, str]], right: list[dict[str, str]], label: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"prediction count mismatch for {label}")
    truth_key = "y_acs" if label == "acs" else "y_omi"
    for a, b in zip(left, right):
        if a["record_id"] != b["record_id"] or a["patient_id"] != b["patient_id"]:
            raise ValueError(f"record/patient mismatch for {label}: {a['record_id']} / {b['record_id']}")
        if a[truth_key] != b[truth_key]:
            raise ValueError(f"label mismatch for {label}: {a['record_id']}")


def acs_arrays(rows: list[dict[str, str]], indices: np.ndarray | None = None):
    selected = rows if indices is None else [rows[int(index)] for index in indices]
    y = np.asarray([int(row["y_acs"]) for row in selected], dtype=np.int64)
    p = np.asarray([[float(row[f"p_{name}"]) for name in ACS_CLASSES] for row in selected])
    return y, p


def omi_arrays(rows: list[dict[str, str]], indices: np.ndarray | None = None):
    selected = rows if indices is None else [rows[int(index)] for index in indices]
    y = np.asarray([int(row["y_omi"]) for row in selected], dtype=np.int64)
    p = np.asarray([float(row["p_omi"]) for row in selected], dtype=float)
    y_acs = np.asarray(
        [int(row.get("y_acs", row.get("y_acs_for_subgroup", -1))) for row in selected],
        dtype=np.int64,
    )
    return y, p, y_acs


def hard_binary(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    sensitivity, specificity = div(tp, tp + fn), div(tn, tn + fp)
    ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    return {
        "sensitivity": sensitivity, "specificity": specificity, "ppv": ppv, "npv": npv,
        "f1": div(2 * ppv * sensitivity, ppv + sensitivity),
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def pooled_acs(rows: list[dict[str, str]]) -> dict:
    y, p = acs_arrays(rows)
    return {"acs": acs_metrics(y, p), "nstemi_vs_ua": nstemi_vs_ua_metrics(y, p)}


def pooled_omi(rows: list[dict[str, str]], thresholds: dict[int, float]) -> dict:
    y, p, y_acs = omi_arrays(rows)
    pred = np.asarray([p[i] >= thresholds[int(row["fold"])] for i, row in enumerate(rows)])
    nste = np.isin(y_acs, [2, 3])
    return {
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "hard_metrics_with_fold_frozen_thresholds": hard_binary(y, pred.astype(np.int64)),
        "nste_auroc": float(roc_auc_score(y[nste], p[nste])),
        "nste_auprc": float(average_precision_score(y[nste], p[nste])),
        "threshold_by_fold": {str(key): float(value) for key, value in thresholds.items()},
    }


def endpoint_acs(rows: list[dict[str, str]], indices: np.ndarray) -> dict[str, float]:
    y, p = acs_arrays(rows, indices)
    one_hot = np.eye(4, dtype=np.int8)[y]
    return {
        "acs_macro_auprc": float(np.mean([
            average_precision_score(one_hot[:, i], p[:, i]) for i in range(4)
        ])),
        "nstemi_auprc": float(average_precision_score((y == 2).astype(np.int8), p[:, 2])),
    }


def endpoint_omi(rows: list[dict[str, str]], indices: np.ndarray) -> dict[str, float]:
    y, p, y_acs = omi_arrays(rows, indices)
    nste = np.isin(y_acs, [2, 3])
    return {
        "omi_auprc": float(average_precision_score(y, p)),
        "omi_nste_auprc": float(average_precision_score(y[nste], p[nste])),
    }


def paired_bootstrap(
    joint_rows: list[dict[str, str]], single_rows: list[dict[str, str]],
    endpoint, replicates: int, seed: int,
) -> dict:
    patient_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(joint_rows):
        patient_indices[row["patient_id"]].append(index)
    patients = np.asarray(sorted(patient_indices), dtype=object)
    full = np.arange(len(joint_rows), dtype=np.int64)
    point_joint, point_single = endpoint(joint_rows, full), endpoint(single_rows, full)
    samples: dict[str, list[float]] = defaultdict(list)
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.asarray(
            [index for patient in sampled for index in patient_indices[str(patient)]], dtype=np.int64
        )
        left, right = endpoint(joint_rows, indices), endpoint(single_rows, indices)
        for key in left:
            samples[key].append(left[key] - right[key])
    output = {}
    for key in point_joint:
        values = np.asarray(samples[key], dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) < int(0.95 * replicates):
            raise RuntimeError(f"too many non-finite bootstrap replicates for {key}")
        low, high = np.percentile(finite, [2.5, 97.5])
        output[key] = {
            "joint": point_joint[key], "single": point_single[key],
            "delta": point_joint[key] - point_single[key],
            "ci95": [float(low), float(high)],
            "ci_crosses_zero": bool(low <= 0 <= high), "finite_replicates": int(len(finite)),
        }
    return output


def format_ms(item: dict[str, float]) -> str:
    return f"{item['mean']:.6f} ± {item['std']:.6f}"


def write_report(path: Path, result: dict) -> None:
    a_joint, a_single = result["fold_mean_std"]["joint_acs"], result["fold_mean_std"]["acs_only"]
    o_joint, o_single = result["fold_mean_std"]["joint_omi"], result["fold_mean_std"]["omi_only"]
    boot_a = result["paired_patient_bootstrap"]["joint_vs_acs_only"]
    boot_o = result["paired_patient_bootstrap"]["joint_vs_omi_only"]
    lines = [
        "# ECG-LFM Joint vs Single-task Nested 5-fold", "",
        "> 所有 checkpoint、threshold 与 calibration 只使用对应 outer-train 内的 inner-validation；每个 outer-test 只评价一次。逐例预测留在受控服务器，不进入 Git。", "",
        "## 1. 五折 mean ± std", "", "### ACS", "",
        "| 方法 | Macro AUROC | Macro AUPRC | Macro F1 | Balanced Accuracy | NSTEMI AUPRC | NSTEMI Sensitivity | NSTEMI-vs-UA AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Joint W0 | {format_ms(a_joint['macro_auroc'])} | {format_ms(a_joint['macro_auprc'])} | {format_ms(a_joint['macro_f1'])} | {format_ms(a_joint['balanced_accuracy'])} | {format_ms(a_joint['nstemi_auprc'])} | {format_ms(a_joint['nstemi_sensitivity'])} | {format_ms(a_joint['nstemi_vs_ua_auprc'])} |",
        f"| ACS-only W0 | {format_ms(a_single['macro_auroc'])} | {format_ms(a_single['macro_auprc'])} | {format_ms(a_single['macro_f1'])} | {format_ms(a_single['balanced_accuracy'])} | {format_ms(a_single['nstemi_auprc'])} | {format_ms(a_single['nstemi_sensitivity'])} | {format_ms(a_single['nstemi_vs_ua_auprc'])} |",
        "", "### OMI", "",
        "| 方法 | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | NPV | NSTE-ACS AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Joint W0 | {format_ms(o_joint['auroc'])} | {format_ms(o_joint['auprc'])} | {format_ms(o_joint['f1'])} | {format_ms(o_joint['sensitivity'])} | {format_ms(o_joint['specificity'])} | {format_ms(o_joint['ppv'])} | {format_ms(o_joint['npv'])} | {format_ms(o_joint['nste_auprc'])} |",
        f"| OMI-only W0 | {format_ms(o_single['auroc'])} | {format_ms(o_single['auprc'])} | {format_ms(o_single['f1'])} | {format_ms(o_single['sensitivity'])} | {format_ms(o_single['specificity'])} | {format_ms(o_single['ppv'])} | {format_ms(o_single['npv'])} | {format_ms(o_single['nste_auprc'])} |",
        "", "## 2. Paired patient bootstrap", "",
        "| 比较 | 指标 | Delta | 95% CI | CI 跨 0 |", "|---|---|---:|---:|---|",
    ]
    for comparison, values in (("Joint − ACS-only", boot_a), ("Joint − OMI-only", boot_o)):
        for metric, item in values.items():
            lines.append(
                f"| {comparison} | {metric} | {item['delta']:.6f} | [{item['ci95'][0]:.6f}, {item['ci95'][1]:.6f}] | {'是' if item['ci_crosses_zero'] else '否'} |"
            )
    lines += [
        "", "## 3. Gate", "",
        f"- positive folds：`{json.dumps(result['positive_fold_counts'], ensure_ascii=False)}`",
        f"- Phase S0 decision：**{result['phase_s0_decision']}**",
        "- PASS 要求三个 primary endpoint 均至少 3/5 folds 同向，且 paired patient bootstrap 95% CI 下界大于 0。",
        "", "## 4. Provenance", "",
        f"- outer records：{result['n_records']}；patients：{result['n_patients']}；fold coverage：100%。",
        "- duplicate/missing/alignment/patient-overlap gate：全部 pass。",
        f"- bootstrap：{result['bootstrap_replicates']} replicates；seed={result['bootstrap_seed']}。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/single_task_vs_joint_evaluation/nested5fold_summary.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("docs/ECG_LFM_joint_vs_single_nested5fold.md"))
    args = parser.parse_args()

    fold_metrics = {"joint_acs": [], "acs_only": [], "joint_omi": [], "omi_only": []}
    joint_rows, acs_rows, omi_rows = [], [], []
    thresholds_joint, thresholds_omi = {}, {}
    hashes, per_fold_rows = [], []
    for fold in range(5):
        paths = {
            "joint": ROOT / f"results/adaptive_weighting_evaluation/w0/fold{fold}",
            "acs": ROOT / f"results/single_task_vs_joint_evaluation/acs_only_w0/fold{fold}",
            "omi": ROOT / f"results/single_task_vs_joint_evaluation/omi_only_w0/fold{fold}",
        }
        metrics = {name: read_json(path / "metrics.json") for name, path in paths.items()}
        alignments = {name: read_json(path / "provenance_alignment_report.json") for name, path in paths.items()}
        if any(item.get("status") != "pass" for item in alignments.values()):
            raise ValueError(f"alignment gate failed in fold {fold}")
        a_joint, a_single = acs_fold(metrics["joint"]), acs_fold(metrics["acs"])
        o_joint, o_single = omi_fold(metrics["joint"]), omi_fold(metrics["omi"])
        fold_metrics["joint_acs"].append(a_joint)
        fold_metrics["acs_only"].append(a_single)
        fold_metrics["joint_omi"].append(o_joint)
        fold_metrics["omi_only"].append(o_single)
        thresholds_joint[fold], thresholds_omi[fold] = o_joint["threshold"], o_single["threshold"]
        for method, values in (
            ("joint_acs", a_joint), ("acs_only", a_single),
            ("joint_omi", o_joint), ("omi_only", o_single),
        ):
            per_fold_rows.append({"fold": fold, "method": method, **values})
        prediction_paths = {
            "joint": paths["joint"] / "outer_test_predictions.csv",
            "acs": paths["acs"] / "outer_test_predictions.csv",
            "omi": paths["omi"] / "outer_test_predictions.csv",
        }
        joint_rows.extend(read_rows(prediction_paths["joint"], fold))
        acs_rows.extend(read_rows(prediction_paths["acs"], fold))
        omi_rows.extend(read_rows(prediction_paths["omi"], fold))
        for path in paths.values():
            for filename in ("metrics.json", "provenance_alignment_report.json"):
                target = path / filename
                hashes.append({"path": str(target.relative_to(ROOT)), "sha256": sha256(target)})
        for path in prediction_paths.values():
            hashes.append({
                "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "protected": True,
            })

    joint_rows.sort(key=lambda row: row["record_id"])
    acs_rows.sort(key=lambda row: row["record_id"])
    omi_rows.sort(key=lambda row: row["record_id"])
    aligned(joint_rows, acs_rows, "acs")
    aligned(joint_rows, omi_rows, "omi")
    if len({row["record_id"] for row in joint_rows}) != len(joint_rows):
        raise ValueError("duplicate record_id across outer folds")
    patients = {row["patient_id"] for row in joint_rows}

    fold_summary = {
        "joint_acs": mean_std(fold_metrics["joint_acs"], ACS_KEYS),
        "acs_only": mean_std(fold_metrics["acs_only"], ACS_KEYS),
        "joint_omi": mean_std(fold_metrics["joint_omi"], OMI_KEYS),
        "omi_only": mean_std(fold_metrics["omi_only"], OMI_KEYS),
    }
    boot_acs = paired_bootstrap(joint_rows, acs_rows, endpoint_acs, args.replicates, args.seed)
    boot_omi = paired_bootstrap(joint_rows, omi_rows, endpoint_omi, args.replicates, args.seed)
    positive = {
        "acs_macro_auprc": sum(
            left["macro_auprc"] > right["macro_auprc"]
            for left, right in zip(fold_metrics["joint_acs"], fold_metrics["acs_only"])
        ),
        "nstemi_auprc": sum(
            left["nstemi_auprc"] > right["nstemi_auprc"]
            for left, right in zip(fold_metrics["joint_acs"], fold_metrics["acs_only"])
        ),
        "omi_auprc": sum(
            left["auprc"] > right["auprc"]
            for left, right in zip(fold_metrics["joint_omi"], fold_metrics["omi_only"])
        ),
    }
    ci_items = {
        "acs_macro_auprc": boot_acs["acs_macro_auprc"],
        "nstemi_auprc": boot_acs["nstemi_auprc"], "omi_auprc": boot_omi["omi_auprc"],
    }
    phase_pass = all(positive[key] >= 3 and ci_items[key]["ci95"][0] > 0 for key in positive)
    result = {
        "status": "pass", "protocol": "patient-level nested outer 5-fold",
        "n_records": len(joint_rows), "n_patients": len(patients),
        "fold_mean_std": fold_summary,
        "pooled_outer_oof": {
            "joint_acs": pooled_acs(joint_rows), "acs_only": pooled_acs(acs_rows),
            "joint_omi": pooled_omi(joint_rows, thresholds_joint),
            "omi_only": pooled_omi(omi_rows, thresholds_omi),
        },
        "paired_patient_bootstrap": {
            "joint_vs_acs_only": boot_acs, "joint_vs_omi_only": boot_omi,
        },
        "positive_fold_counts": positive,
        "phase_s0_decision": "PASS" if phase_pass else "FAIL",
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "artifact_hashes": hashes,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    per_fold = output.with_name("nested5fold_per_fold.csv")
    fields = sorted({key for row in per_fold_rows for key in row}, key=lambda key: (key not in ("fold", "method"), key))
    with per_fold.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_fold_rows)
    write_report(ROOT / args.report, result)
    print(json.dumps({
        "status": "pass", "decision": result["phase_s0_decision"],
        "summary": str(output), "report": str(ROOT / args.report),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
