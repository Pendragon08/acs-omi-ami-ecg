# 实验日志索引与 GitHub 归档策略

**更新时间：** 2026-09-05

## 归档规则

GitHub 保存每折/每 epoch 的脱敏训练损失、验证损失、验证指标、学习率、聚合 metrics、阈值审计、混淆矩阵、ROC/PR、calibration、训练曲线和中文报告。

以下内容不进入 GitHub：原始 ECG、患者标识、patient-level assignments、逐记录 OOF/测试预测、外部平台提交文件、预训练权重、训练 checkpoint、SSH 凭据，以及含患者/凭据的原始日志。Joint-6 总调度日志已公开保存；高频 fairseq event stream 保留在受控服务器，并导出为脱敏逐 epoch CSV。

## 当前实验与证据

| 实验 | 状态 | GitHub 脱敏证据 |
|---|---|---|
| Joint-6 ECG-LFM LoRA | 5/5 完成 | `results/joint6_cv/evaluation/lora/fold*/`、`results/joint6_cv/summary/` |
| Joint-6 ECG-LFM Partial FT | 5/5 完成 | `results/joint6_cv/evaluation/partial/fold*/`、`results/joint6_cv/summary/` |
| Joint-6 五种传统模型 | 25/25 完成 | `results/joint6_cv/evaluation/<model>/fold*/`、`results/joint6_cv/traditional/` |
| 旧 factorized Baseline/LoRA/DoRA 重评估 | 15 checkpoints 完成 | `results/revalidated/` |
| W0–W6 / RDR nested Fold0 | 完成；NO-GO | `results/adaptive_weighting/`、`results/adaptive_weighting_evaluation/` |
| ACS-only / OMI-only / Joint W0 | Fold0 完成 | `results/single_task_vs_joint_evaluation/` |
| 多目标优化 Stage-A | 完成；NO-GO | `results/ecg_lfm_moo/` |
| Direct RL | 完成；NO-GO | `results/ecg_lfm_direct_rl/` |
| R1-v2 | 4/4 完成；未优于 W0 | `results/ecg_lfm_rl_v2/` |
| Cross-task RL v2 | 探索完成；无可信涨点 | `results/ecg_lfm_cross_task_rl_v2/` |
| Joint-7 / Joint-10 | 标签审计后停止 | 对应 `results/joint*_cv/label_audit/` 和专项报告 |
| 历史 OMI/AMI 损失与特征消融 | 已完成 | `results/loss_feature_ablation/` |

当前 Joint-6 的完整总调度日志为 `logs/joint6_cv/full_comparison_gpu1.log`，末尾记录 `complete_all_comparators=true` 和 35/35 folds 完成状态。

## Joint-6 每折归档内容

ECG-LFM LoRA 与本次 Partial FT 的每折目录保存：

- `training_curves.csv`：脱敏 epoch、loss、验证指标和学习率；
- `training_validation_loss_lr.png`：训练/验证损失与 LR；
- `roc_pr_curves.png`；
- `confusion_matrices.png`、`joint6_confusion_matrix.png`；
- `reliability_diagrams.png`；
- 聚合 `metrics.json`、逐类指标、阈值与 calibration 结果。

传统模型目录保存同类聚合 metrics、历史和图表。逐记录 predictions 与 checkpoint 仅保留服务器。

## 复核规则

1. 报告数值必须能追溯到对应 `results/` 文件；validation、outer fold 和历史 external 不混用。
2. 新结果发布前必须检查 duplicate/missing ID、fold coverage、patient overlap、概率归一化和阈值来源。
3. 长任务使用 `screen`、写服务器日志，并以 `CUDA_VISIBLE_DEVICES` 绑定一张 GPU。
4. 当前 Joint-6 是活动任务；旧 factorized、错误多标签定义和旧平台结果必须明确标为历史。
5. 原始日志如需公开，必须先通过 `scripts/export_sanitized_training_logs.py` 脱敏。
