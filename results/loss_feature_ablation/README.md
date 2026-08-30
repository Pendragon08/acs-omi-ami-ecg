# OMI 损失函数与特征消融结果（历史 OMI/AMI 任务）

> **任务定义更正（2026-08-30）**：本结果族针对旧的 OMI/AMI 二分类器，
> 不是当前 ACS `OTHER/STEMI/NSTEMI/UA` 四分类 + OMI 二分类联合任务；保留
> 仅用于方法审计和溯源。

## 中文说明

该结果族遵守固定数据/固定患者级划分协议，包含 Baseline、自动加权 BCE、
Focal、CB-Focal 的五种子比较，以及原始 ECG、年龄、性别、中位数 ECG 的特征
对比。`training_log_sanitized.csv` 是由种子级结果生成的脱敏运行日志（含可用
的最佳 epoch、阈值、loss/指标和 TP/TN/FP/FN），训练曲线和其余汇总文件也已
纳入 GitHub；不保存逐例标识符或预测行。

This result family follows the frozen-data/frozen-split protocol documented in
the report. It includes aggregate five-seed Baseline, automatic weighted BCE,
Focal and class-balanced Focal tables; feature comparisons for raw ECG, age,
sex and median ECG; threshold grids; paired bootstrap output; and selected
figures. The chronological log and audit are kept as Markdown in this folder.

No validation/test record identifiers or prediction rows are committed.
