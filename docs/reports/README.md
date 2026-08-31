# 实验报告索引

所有当前报告优先使用中文。旧独立 sigmoid 任务的报告已从当前分支删除。

## 当前主报告

- [实验结果总报告_最终.md](实验结果总报告_最终.md)：任务、数据、模型对比、
  ECG-LFM Baseline/LoRA/DoRA、外部平台、训练成本和限制；
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
- 脱敏日志索引：`docs/logs/EXPERIMENT_LOG_INDEX.md`。

外部平台没有返回概率或联合三分类混淆矩阵，因此外部 AUROC/AUPRC 和严格的
ACS 三分类总指标目前不可计算。
