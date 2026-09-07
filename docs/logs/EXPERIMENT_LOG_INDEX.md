# 实验日志索引与 GitHub 归档策略

_更新时间：2026-09-07 · 区分已发布的日志与服务器私有日志_

---

## 🔐 归档规则

GitHub 保存每折/每 epoch 的脱敏训练损失、验证损失、验证指标、学习率、聚合 metrics、阈值审计、混淆矩阵、ROC/PR、calibration、训练曲线和中文报告。

以下内容不进入 GitHub：原始 ECG、患者标识、patient-level assignments、逐记录 OOF/测试预测、外部平台提交文件、预训练权重、训练 checkpoint、SSH 凭据，以及含患者/凭据的原始日志。历史 Joint-6 总调度日志和部分 RL 聚合 `train_log.jsonl` 已被 Git 跟踪，继续保留。不能因为 `.log` / `.jsonl` 现已被忽略，就声称它们不在仓库中；也不能直接放行新增原始日志。本轮新实验的高频 fairseq event stream 留在受控服务器。

## 📊 当前实验与证据

| 实验 | 状态 | GitHub 脱敏证据 |
|---|---|---|
| Joint-6 ECG-LFM LoRA | 5/5 完成 | `results/joint6_cv/evaluation/lora/fold*/`、`results/joint6_cv/summary/` |
| Joint-6 ECG-LFM Partial FT | 5/5 完成 | `results/joint6_cv/evaluation/partial/fold*/`、`results/joint6_cv/summary/` |
| Joint-6 五种传统模型 | 25/25 完成 | `results/joint6_cv/traditional/<model>/fold*/` |
| Joint-6 曲线/成本审计修复 | 10 折修复、35/35 复核通过 | `results/joint6_cv/summary/reporting_repair_complete.json`、每折 `reporting_corrections.json` |
| Joint-6 新 LR 对比 | Fold0 两候选训练中，发布时快照 | `results/joint6_optimization_20260907/`、优化报告 |
| 旧 factorized Baseline/LoRA/DoRA 重评估 | 15 checkpoints 完成 | `results/revalidated/` |
| W0–W6 / RDR nested Fold0 | 完成；NO-GO | `results/adaptive_weighting/`、`results/adaptive_weighting_evaluation/` |
| ACS-only / OMI-only / Joint W0 | Fold0 完成 | `results/single_task_vs_joint_evaluation/` |
| 多目标优化 Stage-A | 完成；NO-GO | `results/ecg_lfm_moo/` |
| Direct RL | 完成；NO-GO | `results/ecg_lfm_direct_rl/` |
| R1-v2 | 4/4 完成；未优于 W0 | `results/ecg_lfm_rl_v2/` |
| Cross-task RL v2 | 探索完成；无可信涨点 | `results/ecg_lfm_cross_task_rl_v2/` |
| Joint-7 / Joint-10 | 标签审计后停止 | `results/joint7_stage_a/joint_label_audit/`、`results/joint10_label_audit/` |
| 历史 OMI/AMI 损失与特征消融 | 已完成 | `results/loss_feature_ablation/` |

历史 Joint-6 的[完整总调度日志](../../logs/joint6_cv/full_comparison_gpu1.log)末尾记录 `complete_all_comparators=true` 和 35/35 folds 完成状态。这只证明原模型对比完成，不表示后续优化完成。

## 📦 Joint-6 每折归档内容

ECG-LFM LoRA 与本次 Partial FT 的每折目录保存：

- `training_curves.csv`：脱敏 epoch、loss、验证指标和学习率；
- `training_validation_loss_lr.png`：训练/验证损失与 LR；
- `roc_pr_curves.png`；
- `confusion_matrices.png`、`joint6_confusion_matrix.png`；
- `reliability_diagrams.png`；
- 聚合 `metrics.json`、逐类指标、阈值与 calibration 结果。
- `reporting_corrections.json`：带源 hash 的曲线/参数口径修正；不覆盖原始 metrics 与 receipt。
- `legacy_reporting_v1/`：修复前曲线备份，仅作错误溯源，不作有效训练轨迹。

传统模型目录保存同类聚合 metrics、历史和图表。逐记录 predictions 与 checkpoint 仅保留服务器。

## 🔄 当前优化日志

服务器原始日志：`logs/joint6_optimization_20260907/lr5e5.log` 与 `lr3e5.log`。运行期 checkpoint、raw predictions、`training_telemetry.json` 及首次监测失败尝试仍留服务器；该路径不是可在 GitHub 点击下载的公开文件。

GitHub 已发布：[38 项测试与 CUDA 核验](Joint6_Optimization_Verification_20260907.txt)、[验证摘要](../../results/joint6_optimization_20260907/verification_summary.json)及[优化报告](../ECG_LFM_Joint6_Optimization_Report.md)。候选结束并通过审阅后再同步安全曲线和聚合结果，不上传整个训练目录。

## 🔍 复核规则

1. 报告数值必须能追溯到对应 `results/` 文件；validation、outer fold 和历史 external 不混用。
2. 新结果发布前必须检查 duplicate/missing ID、fold coverage、patient overlap、概率归一化和阈值来源。
3. 长任务使用 `screen`、写服务器日志，并以 `CUDA_VISIBLE_DEVICES` 绑定一张 GPU。
4. 当前 Joint-6 是活动任务；旧 factorized、错误多标签定义和旧平台结果必须明确标为历史。
5. 原始日志如需公开，必须使用字段白名单导出并再次审阅。旧任务通用导出器为 `scripts/export_sanitized_training_logs.py`；当前 Joint-6 必须使用 `experiments/ecg_lfm/training_reporting.py` 的单折 epoch-end 解析，避免重新引入重复日志或跨折曲线。
