# Joint-6 分阶段优化结果

_发布快照：2026-09-07 11:54 北京时间 · 本阶段尚未完成_

---

## 🔄 当前状态

报告修复完成，38 项测试通过。LR=5e-5（GPU0）与 LR=3e-5（GPU1）两个 Fold0 候选仍在训练。静态加权、阈值/校准、追加五折及多 seed 尚未启动。完整执行约束、监测修复重启原因和晋级条件见[优化报告](../../docs/ECG_LFM_Joint6_Optimization_Report.md)。

## 📦 已发布产物

| 文件 | 含义 |
|---|---|
| [lr_inner_comparison.csv](lr_inner_comparison.csv) | 学习率对比的 inner-validation 聚合表；未完成值不填成绩 |
| [lr_selection.json](lr_selection.json) | 筛选门控状态，当前 pending |
| [baseline_lr1e4/inner_metrics.json](baseline_lr1e4/inner_metrics.json) | 复用并核验过的历史基线 inner 指标 |
| [baseline_lr1e4/inner_evaluation_complete.json](baseline_lr1e4/inner_evaluation_complete.json) | 基线推理产物/代码 hash 凭据 |
| [verification_summary.json](verification_summary.json) | 38 项测试、10 折修复与 35/35 复核摘要 |

## 🔐 解释与保留规则

本目录的基线 Joint-6 AUPRC=0.394075、OMI AUPRC=0.432049 是 **inner-validation** 指标，不是五折 outer 结果。只有全部 LR 候选完成且安全门控通过后才能做候选结论。GitHub 更新不等于实时监控，不得根据此快照认定服务器永久处于该状态。

训练目录、首次监测失败尝试、checkpoint、原始日志、患者级预测不上传。后续仅将审阅后的聚合指标、曲线、图表和配置摘要追加到本目录。旧五折主结果独立保存在 [joint6_cv/](../joint6_cv/README.md)。
