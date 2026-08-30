# ECG-LFM 校正联合任务（v2）

结果总表见 [`docs/reports/实验结果总报告_最终.md`](../../docs/reports/实验结果总报告_最终.md)。

该目录用于保存校正后的 ECG-LFM Baseline/LoRA/DoRA 结果。当前正式任务是
ACS `OTHER/STEMI/NSTEMI/UA` 互斥单标签四分类 + 独立 OMI/NOMI 二分类，
不是四个独立 sigmoid 标签。

Baseline、LoRA、DoRA 五折均已完成，并已归档各自的 `summary/summary.json`、阈值分析、
ROC/PR 曲线、混淆矩阵、训练/验证曲线、脱敏 epoch 日志和外部平台 hard-label JSON。
DoRA fold3 的中断/恢复及 fold4 的 GPU1 并行过程均有日志索引记录。

## 目录约定

- `baseline/`：ECG-LFM 全参数微调；
- `lora/`：rank=8、alpha=16 的 LoRA 微调；
- `dora/`：rank=8、alpha=16 的 DoRA 微调；
- 每个已完成方法的 `summary/` 保存 pooled OOF 指标、阈值、曲线和混淆矩阵；
- 每个 `fold*/training_log_sanitized.csv` 只含 epoch、loss、验证指标和学习率。
- DoRA 的 fold4 训练目录曾位于并行结果根目录，复制前后 checkpoint SHA-256 一致；
  正式结果只引用 `dora/` 下的五个 fold。

三种方式的参数量、显存和 wall-clock 训练成本对比已记录在
`docs/reports/corrected_ecg_lfm_joint_20260829.md` 的“训练成本对比”章节。

脱敏日志只包含 phase、epoch、loss、准确率、学习率和最佳验证损失，不含原始
Fairseq JSONL、服务器路径、患者/记录标识、ECG 波形、checkpoint 或逐例预测。
