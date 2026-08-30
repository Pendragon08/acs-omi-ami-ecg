# ECG-LFM DoRA 校正联合任务

本目录保存 DoRA（rank=8、alpha=16、dropout=0.05）在校正任务上的五折结果。
任务为 ACS `OTHER/STEMI/NSTEMI/UA` 互斥四分类 + 独立 OMI/NOMI 二分类，
不是四个独立 sigmoid。ACS 使用 softmax/argmax，OMI 阈值由训练记录的 pooled
OOF 验证结果选择；外部平台提交固定使用 OOF Youden 阈值 `0.05`。

## 已归档文件

- `summary/summary.json`、`threshold_analysis.json`：pooled OOF 指标、阈值和
  manifest/checkpoint 哈希；
- `summary/training_curves.csv`、`training_validation_loss_lr.png`：训练/验证损失
  与学习率；
- `summary/roc_pr_curves.png`、`confusion_matrices.png`：ROC/PR 和混淆矩阵；
- `fold0`--`fold4/training_log_sanitized.csv`：脱敏逐 epoch 日志；
- `official_platform_4label_metrics.json`：外部平台返回的四标签 hard-label 指标
  （提交 CSV 本身不在仓库）。

fold3 曾在训练中断后恢复，fold4 在 GPU1 并行完成后复制到本目录；复制前后
checkpoint SHA-256 一致。原始 ECG、患者级表格、checkpoint、逐例概率和原始
Fairseq JSONL 均不上传。
