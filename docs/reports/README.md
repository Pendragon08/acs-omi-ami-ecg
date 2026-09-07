# 实验报告索引

_更新时间：2026-09-07 · 上级入口：[中文文档导航](../README.md)_

---

## 📊 当前总报告

- [实验结果总报告_最终.md](实验结果总报告_最终.md)：当前任务、35/35 folds Joint-6 对比、逐类指标、训练成本、历史消融、MOO/RL 结论、外部测试边界与下一步。
- [../ECG_LFM_Joint6_CV_Comparison_Report.md](../ECG_LFM_Joint6_CV_Comparison_Report.md)：Joint-6 七种模型完整五折报告。
- [../ECG_LFM_Joint6_Evaluation_Report.md](../ECG_LFM_Joint6_Evaluation_Report.md)：Joint-6 evaluator、概率恢复与逐类评价说明。
- [实验完整性审计_20260905.md](实验完整性审计_20260905.md)：35/35 folds、结果文件、声明边界与 GitHub 安全归档复核。

## 🔄 当前优化与修订

- [分阶段优化报告](../ECG_LFM_Joint6_Optimization_Report.md)：报告修复已完成；Fold0 学习率实验在研，后续加权/校准/多折/多 seed 尚未启动。
- [仓库整理记录](仓库整理记录_20260907.md)：本次发布内容与历史保留范围。
- [修正训练成本](../../results/joint6_cv/summary/training_cost.csv)：有效训练参数、显存和时间测量范围。2026-09-05 历史审计中的旧成本表述以本次修订为准。

## 📚 历史专项实验报告

- [../ECG_LFM_corrected_evaluation_report.md](../ECG_LFM_corrected_evaluation_report.md)：旧 factorized Baseline/LoRA/DoRA 15 个 checkpoint 的 record-ID 修正重评估；
- [../ECG_LFM_adaptive_weighting_experiment_report.md](../ECG_LFM_adaptive_weighting_experiment_report.md)：W0–W6 静态权重与 RDR；
- [../ECG_LFM_multi_objective_optimization_report.md](../ECG_LFM_multi_objective_optimization_report.md)：多目标优化 Stage-A；
- [../ECG_LFM_direct_RL_post_training_report.md](../ECG_LFM_direct_RL_post_training_report.md)：Direct RL；
- [../ECG_LFM_R1_v2_stabilization_report.md](../ECG_LFM_R1_v2_stabilization_report.md)：R1-v2 稳定化；
- [../ECG_LFM_CrossTask_RL_v2_report.md](../ECG_LFM_CrossTask_RL_v2_report.md)：Cross-task RL v2；
- [../ECG_LFM_joint7_label_audit_no_go_report.md](../ECG_LFM_joint7_label_audit_no_go_report.md)：Joint-7 标签硬门控；
- [../ECG_LFM_Nature_Joint10_CV_Comparison_Report.md](../ECG_LFM_Nature_Joint10_CV_Comparison_Report.md)：Joint-10 标签硬门控；
- [loss_feature_ablation/](loss_feature_ablation/)：旧 OMI/AMI 二分类损失和特征消融，仅作历史方法记录。

## 🔗 证据目录

- 当前 Joint-6：`results/joint6_cv/`；
- 当前在研优化：`results/joint6_optimization_20260907/`；
- 修正旧双 Head：`results/revalidated/`；
- W0–W6：`results/adaptive_weighting/`、`results/adaptive_weighting_evaluation/`；
- 单任务对照：`results/single_task_vs_joint_evaluation/`；
- MOO/RL：`results/ecg_lfm_moo/`、`results/ecg_lfm_direct_rl/`、`results/ecg_lfm_rl_v2/`、`results/ecg_lfm_cross_task_rl_v2/`；
- 脱敏日志说明：[../logs/EXPERIMENT_LOG_INDEX.md](../logs/EXPERIMENT_LOG_INDEX.md)。

当前 Joint-6 尚无独立外部测试。旧 external/platform 结果使用不同端点或模型选择流程，只能作为历史工程证据。
