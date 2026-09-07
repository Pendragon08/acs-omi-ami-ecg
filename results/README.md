# 聚合实验产物

_更新时间：2026-09-07 · 只归档安全聚合结果_

---

本目录只保存可公开复核的聚合结果、脱敏逐 epoch 曲线、可视化图和 provenance 摘要。原始 ECG、患者级表格、checkpoint、逐记录预测、外部提交文件和原始服务器日志不进入 GitHub。

## 📊 当前主结果：Joint-6

`joint6_cv/` 已完成 7 个模型的 patient-level nested 5-fold（35/35 folds）：

- ECG-LFM LoRA；
- ECG-LFM Partial FT（修正后有效参与更新参数占 99.936%，不能仅凭 `requires_grad=100%` 称为全参微调）；
- InceptionTime；
- ResNet1D_Wang；
- LSTM；
- Transformer1D；
- ViT。

目录内容：

- `label_audit/`：不含患者标识的标签分布与划分审计；
- `evaluation/<method>/fold*/`：每折聚合 metrics、confusion matrix、calibration、阈值审计、脱敏训练曲线和图；
- `summary/`：fold-wise、mean/std、pooled OOF 聚合结果、逐类指标和训练成本；
- `traditional/`：传统模型训练过程聚合产物。

10 个 ECG-LFM 折的曲线和有效参数统计于 2026-09-07 修正；旧指标/预测不变，旧曲线仅保存在 `legacy_reporting_v1/` 供审计。入口：[Joint-6 目录说明](joint6_cv/README.md)。

当前最佳 ECG-LFM LoRA：Joint-6 Macro AUPRC 0.3843 ± 0.0056、ACS Macro AUPRC 0.5445 ± 0.0067、OMI AUPRC 0.4299 ± 0.0258。

## 🔄 在研优化

[joint6_optimization_20260907/](joint6_optimization_20260907/README.md) 保存当前分阶段优化的发布快照。报告修复已完成；两个新 LR 的 Fold0 候选在本次核验时仍处于训练。基线 `inner_metrics.json` 仅是 inner-validation 筛选结果，不可与上述五折 outer 指标混用。

## 📚 历史实验目录

- `revalidated/`：旧 factorized Baseline/LoRA/DoRA 的修正 pooled OOF；
- `nested_validation/`：patient-level nested split 的安全聚合 provenance；
- `adaptive_weighting/`、`adaptive_weighting_evaluation/`：W0–W6 静态权重与 RDR Fold0；
- `single_task_vs_joint_evaluation/`：ACS-only、OMI-only 与 Joint W0 Fold0；
- `ecg_lfm_moo/`：多目标优化 Stage-A；
- `ecg_lfm_direct_rl/`、`ecg_lfm_rl_v2/`、`ecg_lfm_cross_task_rl_v2/`：RL 探索结果；
- `loss_feature_ablation/`：历史 OMI/AMI 损失和输入特征消融，端点与当前任务不同；
- `corrected_*`：旧 corrected factorized 工程结果，只作历史参照。

统一中文解释见 [实验结果总报告](../docs/reports/实验结果总报告_最终.md)，当前 Joint-6 完整报告见 [ECG_LFM_Joint6_CV_Comparison_Report.md](../docs/ECG_LFM_Joint6_CV_Comparison_Report.md)。
