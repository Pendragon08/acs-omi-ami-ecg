# 实验报告索引

所有当前报告优先使用中文。旧独立 sigmoid 任务的报告已从当前分支删除。

## 当前主报告

- [实验结果总报告_最终.md](实验结果总报告_最终.md)：2026-09-01 更新的正式总报告；包含
  修正五折重评估、nested Fold0 W0–W6、W0 校准、单任务/联合学习、外部平台与训练成本；
- [../ECG_LFM_adaptive_weighting_experiment_report.md](../ECG_LFM_adaptive_weighting_experiment_report.md)：
  自适应类别重加权专项报告与 RDR NO-GO 依据；
- [../ECG_LFM_corrected_evaluation_report.md](../ECG_LFM_corrected_evaluation_report.md)：
  Baseline/LoRA/DoRA 15 个旧 checkpoint 的 record-ID 修正重评估；
- [实验完整性审计_20260831.md](实验完整性审计_20260831.md)：代码与结果完整性复核、
  清理范围和剩余限制；
- [../plans/CORRECTED_TASK_FORMULATION.md](../plans/CORRECTED_TASK_FORMULATION.md)：
  当前唯一任务规范。

## 有效历史消融

`loss_feature_ablation/` 保存 OMI/AMI 二分类阶段完成的损失和输入特征消融，
包括 CB-Focal、年龄、性别、中位数 ECG 等实验。它们使用不同端点，只能作为
历史方法学记录，不能替代当前 ACS 亚型 + OMI 联合结果。

## 结果证据

- ECG-LFM：`results/corrected_ecg_lfm_joint_bs64_v2/`；
- E4：`results/corrected_joint_acs_omi_e4/`；
- 传统模型：`results/corrected_joint_model_comparison/`；
- nested W0–W6：`results/adaptive_weighting/`、`results/adaptive_weighting_evaluation/`；
- 单任务与联合对照：`results/single_task_vs_joint_evaluation/`；
- 脱敏日志索引：`docs/logs/EXPERIMENT_LOG_INDEX.md`。

外部平台没有返回概率或联合四状态混淆矩阵，因此外部 AUROC/AUPRC 和严格的
ACS 四状态总指标目前不可计算。本轮 nested Fold0 结果尚未重新提交该平台。
