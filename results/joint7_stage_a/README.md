# Joint-7 Fold0 Stage-A

状态：**训练前数据硬门未通过，未启动训练**。

真实 17,960 条有标签训练池中，`OTHER+OMI=1` 符合预设 singleton 处理规则，但 `UA+OMI=0`。所给方案要求任一主要联合类在 inner-train 中 `N=0` 时立即停止，因此没有生成 Joint-7 checkpoint、预测、模型指标或图表，也未进行 outer/external 模型推理。

本目录仅保存可公开的聚合证据：

- `joint_label_audit/audit_summary.json`：审计状态、hash 与 provenance；
- `joint_label_audit/overall_joint_counts.csv`：原始八组合的 ECG/患者聚合计数；
- `joint_label_audit/fold_joint_counts.csv`：Fold0 inner-train/inner-validation/outer-test 七联合类聚合计数；
- `joint_label_audit/patient_joint_counts.csv`：各组合的唯一患者数（不含患者 ID）；
- `stage_a_not_run.json`：未运行原因。

包含 record_id/patient_id 的 `exceptions.csv` 只保留在受控服务器，不进入 Git。

完整中文结论见仓库根目录 `ECG_LFM_joint_acs_omi_stage_a_report.md`。
