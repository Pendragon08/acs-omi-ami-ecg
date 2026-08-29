# ECG-LFM 校正联合任务（v2）

该目录用于保存校正后的 ECG-LFM Baseline/LoRA/DoRA 结果。队列在 GPU1
运行中，当前只归档了 Baseline fold0 已完成 epoch 的脱敏过程日志：
`baseline/fold0/training_log_sanitized.csv`。它是中间运行证据，不是最终性能
指标；15 个运行全部完成并统一评估后才会添加 summary、ROC/PR、混淆矩阵、
五折指标和外部平台结果。

脱敏日志只包含 phase、epoch、loss、准确率、学习率和最佳验证损失，不含原始
Fairseq JSONL、服务器路径、患者/记录标识、ECG 波形、checkpoint 或逐例预测。
