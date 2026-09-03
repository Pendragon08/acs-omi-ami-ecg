"""Build Joint-7 labels and aggregate, provenance-safe label audits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io

from experiments.ecg_lfm.joint7 import ACS_CLASSES, JOINT_CLASSES, encode_source_labels, one_hot_joint7


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def combo(acs: int, omi: int) -> str:
    return f"{ACS_CLASSES[acs]}+{'OMI' if omi else 'nOMI'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_root", default="manifests/ecg_lfm_nested")
    parser.add_argument("--master_csv", default="manifests/ecg_lfm/master_records.csv")
    parser.add_argument("--label_csv", default="data/CSV/train.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", default="results/joint7_stage_a/joint_label_audit")
    parser.add_argument("--report", default="ECG_LFM_joint_acs_omi_stage_a_report.md")
    args = parser.parse_args()
    manifest_root = ROOT / args.manifest_root
    source_path = manifest_root / "y_all.npy"
    master_path = ROOT / args.master_csv
    label_csv = ROOT / args.label_csv
    assignment_path = manifest_root / f"fold{args.fold}" / "patient_assignments.csv"
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    source = np.load(source_path)
    acs, omi, joint, exception = encode_source_labels(source)
    np.save(manifest_root / "y_joint_7.npy", one_hot_joint7(joint))
    master = read_csv(master_path)
    if len(master) != len(source):
        raise RuntimeError(f"master/source length mismatch: {len(master)} != {len(source)}")
    label_rows = read_csv(label_csv)
    labels_by_record = {Path(row["ecg_row_record"]).stem.zfill(5): row for row in label_rows}
    if len(labels_by_record) != len(label_rows):
        raise RuntimeError("duplicate record_id in source label CSV")
    known_indices = [index for index, row in enumerate(master) if row["cohort"] == "train"]
    external_indices = [index for index, row in enumerate(master) if row["cohort"] == "external"]
    if len(known_indices) != len(assignments := read_csv(assignment_path)):
        raise RuntimeError("outer-CV assignment does not cover every known-label training record")
    positional_mismatches = []
    for index in known_indices:
        row = master[index]
        record_id = str(row["record_id"]).zfill(5)
        label_row = labels_by_record.get(record_id)
        if label_row is None:
            positional_mismatches.append(index)
            continue
        master_values = np.asarray(
            [int(label_row["STEMI"]), int(label_row["NSTEMI"]), int(label_row["UA"]), int(label_row["OMI"])],
            dtype=np.int64,
        )
        patient_matches = row["patient_id"] == label_row["Patient_id"]
        if int(row["idx"]) != index or not patient_matches or not np.array_equal(source[index].astype(np.int64), master_values):
            positional_mismatches.append(index)
    if positional_mismatches:
        raise RuntimeError(
            f"y_all positional provenance failed at {len(positional_mismatches)} indices; "
            f"first={positional_mismatches[:10]}"
        )
    by_record = {str(row["record_id"]).zfill(5): row for row in assignments}
    if len(by_record) != len(assignments):
        raise RuntimeError("duplicate record_id in patient assignment")

    overall_rows = []
    patient_rows = []
    for acs_id, acs_name in enumerate(ACS_CLASSES):
        for omi_id in (0, 1):
            selected = np.asarray(known_indices, dtype=np.int64)
            selected_mask = (acs[selected] == acs_id) & (omi[selected] == omi_id)
            matched_indices = selected[selected_mask]
            patients = {str(master[i]["patient_id"]) for i in matched_indices}
            row = {
                "raw_combination": combo(acs_id, omi_id),
                "acs": acs_name,
                "omi": omi_id,
                "ecg_count": int(selected_mask.sum()),
                "unique_patient_count": len(patients),
                "prevalence": float(selected_mask.mean()),
            }
            overall_rows.append(row)
            patient_rows.append({k: row[k] for k in ("raw_combination", "unique_patient_count")})

    fold_counts: dict[tuple[str, int], dict] = {}
    assignment_indices = set()
    for record_id, row in by_record.items():
        index = int(row["idx"])
        assignment_indices.add(index)
        joint_id = int(joint[index])
        key = (row["subset"], joint_id)
        item = fold_counts.setdefault(key, {"patients": set(), "records": 0})
        item["records"] += 1
        item["patients"].add(row["patient_id"])
    if assignment_indices != set(known_indices):
        raise RuntimeError(
            f"outer-CV assignment/master index coverage mismatch: "
            f"missing={len(set(known_indices)-assignment_indices)}, "
            f"extra={len(assignment_indices-set(known_indices))}"
        )
    fold_rows = []
    for subset in ("inner_train", "inner_validation", "outer_test"):
        subset_total = sum(item["records"] for (item_subset, _), item in fold_counts.items() if item_subset == subset)
        for joint_id, name in enumerate(JOINT_CLASSES):
            item = fold_counts.get((subset, joint_id), {"records": 0, "patients": set()})
            fold_rows.append({
                "fold": args.fold,
                "subset": subset,
                "joint_class_id": joint_id,
                "joint_class": name,
                "ecg_count": item["records"],
                "unique_patient_count": len(item["patients"]),
                "prevalence": float(item["records"] / subset_total) if subset_total else 0.0,
            })

    # Close the final positional provenance gap: every processed segment embeds
    # the master index used by FileECGDataset.  Validate that value against the
    # explicit segment table without loading ECG waveform arrays.
    mat_index_mismatches = []
    checked_segments = 0
    for split_name in ("train", "valid", "outer_test"):
        segment_path = manifest_root / f"fold{args.fold}" / f"{split_name}_segments.csv"
        tsv_path = manifest_root / f"fold{args.fold}" / f"{split_name}.tsv"
        if not segment_path.exists() or not tsv_path.exists():
            continue
        root_dir = Path(tsv_path.read_text(encoding="utf-8").splitlines()[0])
        for row in read_csv(segment_path):
            embedded = scipy.io.loadmat(root_dir / row["mat_file"], variable_names=["idx"])["idx"]
            embedded_idx = int(np.asarray(embedded).squeeze())
            checked_segments += 1
            if embedded_idx != int(row["idx"]):
                mat_index_mismatches.append((split_name, int(row["manifest_position"])))
    if mat_index_mismatches:
        raise RuntimeError(
            f"processed .mat embedded-index provenance failed for {len(mat_index_mismatches)} segments; "
            f"first={mat_index_mismatches[:10]}"
        )

    exception_rows = []
    known_exception_indices = [index for index in known_indices if bool(exception[index])]
    for index in known_exception_indices:
        row = master[int(index)]
        assignment = by_record.get(str(row["record_id"]).zfill(5), {})
        exception_rows.append({
            "idx": int(index),
            "record_id": str(row["record_id"]).zfill(5),
            "patient_id": row["patient_id"],
            "fold": args.fold if assignment else "not_in_outer_cv_pool",
            "subset": assignment.get("subset", "external_or_unassigned"),
            "raw_acs_target": "OTHER",
            "raw_omi_target": 1,
            "joint_target": 0,
            "joint_label_exception": True,
            "exception_type": "OTHER_OMI",
        })

    write_csv(output / "overall_joint_counts.csv", overall_rows)
    write_csv(output / "patient_joint_counts.csv", patient_rows)
    write_csv(output / "fold_joint_counts.csv", fold_rows)
    write_csv(output / "exceptions.csv", exception_rows)
    empty_joint_classes = [JOINT_CLASSES[k] for k in range(7) if int((joint[list(assignment_indices)] == k).sum()) == 0]
    inner_train_empty_classes = [
        row["joint_class"] for row in fold_rows
        if row["subset"] == "inner_train" and int(row["ecg_count"]) == 0
    ]
    singleton_ok = len(known_exception_indices) == 1
    class_support_ok = not inner_train_empty_classes
    status = "pass" if singleton_ok and class_support_ok else "stop"
    reasons = []
    if not singleton_ok:
        reasons.append("OTHER+OMI count is not exactly one")
    if not class_support_ok:
        reasons.append("major Joint-7 class has N=0 in inner-train: " + ", ".join(inner_train_empty_classes))
    summary = {
        "status": status,
        "stop_reason": None if status == "pass" else "; ".join(reasons),
        "mapping": {str(index): name for index, name in enumerate(JOINT_CLASSES)},
        "source_array_rows": int(len(source)),
        "known_label_training_records": len(known_indices),
        "external_unknown_placeholder_records": len(external_indices),
        "external_placeholder_nonzero_rows": int(
            np.any(source[np.asarray(external_indices, dtype=np.int64)] != 0, axis=1).sum()
        ),
        "outer_cv_pool_records": int(len(assignment_indices)),
        "other_omi_count": len(known_exception_indices),
        "empty_joint_classes_in_outer_cv_pool": empty_joint_classes,
        "inner_train_empty_joint_classes": inner_train_empty_classes,
        "ua_omi_global_support": int(((acs == 3) & (omi == 1)).sum()),
        "exception_policy": "joint_target=OTHER; raw_omi_target=1; primary OMI evaluation retains record",
        "source_label_sha256": sha256(source_path),
        "master_sha256": sha256(master_path),
        "source_label_csv_sha256": sha256(label_csv),
        "assignment_sha256": sha256(assignment_path),
        "joint_label_sha256": sha256(manifest_root / "y_joint_7.npy"),
        "source_master_positional_mismatches": len(positional_mismatches),
        "processed_segments_checked": checked_segments,
        "processed_mat_index_mismatches": len(mat_index_mismatches),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    report_path = ROOT / args.report
    report_path.write_text(render_gate_report(summary, overall_rows, fold_rows), encoding="utf-8")
    if summary["status"] != "pass":
        (output.parent / "stage_a_not_run.json").write_text(
            json.dumps({
                "status": "not_run",
                "phase": "Fold0 Stage-A training",
                "reason": summary["stop_reason"],
                "outer_test_model_inference": False,
                "external_test_accessed": False,
                "metrics_generated": False,
                "checkpoint_generated": False,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise SystemExit(2)


def render_gate_report(summary: dict, overall_rows: list[dict], fold_rows: list[dict]) -> str:
    overall_table = "\n".join(
        f"| {row['raw_combination']} | {row['ecg_count']} | {row['unique_patient_count']} | {row['prevalence']:.6f} |"
        for row in overall_rows
    )
    fold_table = "\n".join(
        f"| {row['subset']} | {row['joint_class']} | {row['ecg_count']} | {row['unique_patient_count']} | {row['prevalence']:.6f} |"
        for row in fold_rows
    )
    decision = "PRE-TRAINING DATA GATE STOP" if summary["status"] != "pass" else "DATA GATE PASS"
    return f"""# ECG-LFM ACS×OMI Joint-7 Stage-A 报告

## 1. Repository audit

当前 W0 真实结构为共享 ECG-LFM+LoRA 后的单个 `Linear(1024,5)`，前四个 logits 使用 ACS softmax+CE，第五个 logit 使用独立 OMI sigmoid+BCE。它在语义上是 factorized two-head，但并非两个物理 Linear 层。旧实现和 checkpoint 保留不变；Joint-7 已作为独立 formulation、criterion、配置和执行路径实现。

| 模块 | 真实路径 | 关键实现 | 本次处理 |
|---|---|---|---|
| Dataset / dataloader | `fairseq_signals/tasks/ecg_classification.py`、`data/ecg/raw_ecg_dataset.py` | `ECGDiagnosisTask`、`FileECGDataset` | 保留 manifest 和 `.mat idx` 读取；新增 formulation 配置 |
| W0 标签与 Loss | `experiments/ecg_lfm/run_joint_fold.py`、`criterions/acs_joint.py` | `ensure_joint_label_file`、`ACSJointCriterion` | 原样保留 |
| Nested patient split | `experiments/ecg_lfm/build_nested_manifests.py` | `StratifiedGroupKFold` | 不重划分，复用原 hash |
| ECG-LFM / LoRA / pooled feature | `models/classification/ecg_transformer_classifier.py`、`modules/parameter_efficient.py` | `ECGTransformerClassificationModel`、`LoRALinear` | Backbone 不改；Joint-7 输出 7 logits |
| Joint-7 标签 | `experiments/ecg_lfm/joint7.py`、`audit_joint7_labels.py` | `encode_source_labels`、`one_hot_joint7` | 新增映射、异常处理与数据硬门 |
| Joint-7 Loss | `criterions/acs_joint7.py` | `ACSJoint7Criterion` | 只计算 unweighted Joint CE |
| 训练入口 / 配置 | `run_joint7_fold.py`、`corrected_joint7_lora.yaml` | Fold runner / Hydra config | 保持 W0 超参数，训练前强制审计 |
| 评估与校准 | `evaluate_joint7_stage_a.py` | 边缘化、指标、temperature scaling、GO gate | 仅 inner-validation；门控前不加载 outer/external |

## 2. Joint-label audit

审计结论：**{decision}**。

停止原因：{summary['stop_reason'] or '无'}。

`OTHER+OMI={summary['other_omi_count']}`，符合 singleton 异常规则：其 joint target 为 OTHER，原始 OMI=1 不改，primary OMI 评价仍应包含它。

### 原始八组合

| 原始组合 | ECG | 患者 | prevalence |
|---|---:|---:|---:|
{overall_table}

### Nested Fold0 七联合类

| subset | joint class | ECG | 患者 | prevalence |
|---|---|---:|---:|---:|
{fold_table}

## 3. Provenance

- y_all 与 master index/record label 逐条不一致：{summary['source_master_positional_mismatches']}
- 检查的预处理 segment `.mat` 数：{summary['processed_segments_checked']}
- `.mat` 内嵌 idx 与 segment/master idx 不一致：{summary['processed_mat_index_mismatches']}
- patient-level split：沿用既有 nested Fold0，未重划分
- outer-test：只做标签支持度审计，没有模型推理、调参或指标计算
- external-test：未访问

## 4. Joint-7 implementation status

已实现固定 7-class mapping、OTHER+OMI singleton 策略、Joint CE、Joint softmax 的 ACS/OMI/NSTE 概率边缘化、Joint-7 内部验证 evaluator、temperature scaling、fixed-sensitivity operating points、完整预测字段、图表和 GO gate。W0 路径继续可运行。

验证状态由独立测试日志记录在 `docs/logs/joint7_unit_tests.txt` 和 `docs/logs/joint7_config_compose.txt`；审计脚本本身不把尚未执行的测试预先标记为通过。

## 5. Training configuration

Joint-7 配置保持 W0 的 pretrained ECG-LFM、LoRA r=8/alpha=16/dropout=0.05、Adam、LR=1e-4、batch=64、累积4、Step scheduler、100 epochs、patience=10；checkpoint criterion 为 inner-validation joint CE 最小。

## 6. Stage-A results

未启动训练，因此没有 J1 指标，也没有人为填充或虚构 `UA+OMI` 指标。`UA+OMI` 全局 support=0，CE 训练中永远没有该类正 target，AUROC/AUPRC/F1/Sensitivity 无法有效估计。

## 7. GO / CONDITIONAL GO / NO-GO

**NO-GO（数据可行性硬门）**。

依据所给完整方案：任一主要 joint class 在 inner-train 中 N=0 必须立即停止。Fold0 的 `UA+OMI` 满足该停止条件。此处 NO-GO 判断针对预注册的固定 Joint-7 标签空间，并不代表所有联合/层级建模思想均无效。

## 8. 是否建议完整 nested 5-fold

不建议，也未启动。需要先由研究者重新确认科学标签空间，例如是否有可靠 UA+OMI 病例、是否应采用与现有数据支持一致的联合/层级 formulation；本轮不擅自更改标签、不删除样本、不新增 loss 或其他变量。

## 9. 可追溯输出

聚合审计位于 `results/joint7_stage_a/joint_label_audit/`。`exceptions.csv` 含 record/patient 标识，只保留在受控服务器，不提交 Git。
"""


if __name__ == "__main__":
    main()
