# Joint-6 五折对比结果说明

_当前已完成主结果 · 7 个模型 × 5 folds · 2026-09-07 报告口径修订_

---

## 📊 查看结果

完整分析见[中文总报告](../../docs/reports/实验结果总报告_最终.md)、[五折对比报告](../../docs/ECG_LFM_Joint6_CV_Comparison_Report.md)及[评价报告](../../docs/ECG_LFM_Joint6_Evaluation_Report.md)。当前主要排序指标为 Joint-6 Macro AUPRC。

| 目录 | 内容 |
|---|---|
| [summary/](summary/) | fold-wise、mean/std、pooled 汇总、逐类指标、成本与修复凭据 |
| [evaluation/lora/](evaluation/lora/) | ECG-LFM LoRA Fold0–4 指标和图表 |
| [evaluation/partial/](evaluation/partial/) | ECG-LFM Partial FT Fold0–4 指标和图表 |
| [traditional/](traditional/) | 五个传统模型逐折指标、训练轨迹和图表 |
| [label_audit/](label_audit/) | 不含患者 ID 的类别计数及划分审计摘要 |

## 🔍 2026-09-07 修正

10 个 ECG-LFM 折的曲线已按真实单折、最终完整运行和 epoch-end 记录重新导出。每折 `reporting_corrections.json` 绑定原始凭据与修正统计；`legacy_reporting_v1/` 保存有误的旧曲线，仅供审计。**不得再次使用旧曲线绘图或汇总。**

[corrected_training_statistics.csv](summary/corrected_training_statistics.csv)区分 `requires_grad` 与有效参与更新的参数。Partial FT 有效比例为 99.936%，LoRA 为 0.939%。[training_cost.csv](summary/training_cost.csv)保留显存/时间采集范围；整卡采样值不能当作 PyTorch allocated 峰值。

[reporting_repair_complete.json](summary/reporting_repair_complete.json)记录修复凭据。原始 `metrics.json`、预测和评价 receipt 未改写；本次修复没有产生新的性能提升。

## 🔐 评价与发布边界

这里的五折结果属于内部 outer-fold 评价，不是独立外部平台测试。inner-validation 负责 checkpoint 和 OMI Youden 阈值选择。逐记录预测、患者划分表与 checkpoint 保存在受控服务器，不在本目录公开发布。

后续优化另见[在研目录](../joint6_optimization_20260907/README.md)，不得覆盖本批次历史结果。
