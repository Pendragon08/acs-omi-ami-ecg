# ECG-LFM 校正联合任务（v2）

该目录用于保存校正后的 ECG-LFM Baseline/LoRA/DoRA 结果。当前正式任务是
ACS `OTHER/STEMI/NSTEMI/UA` 互斥单标签四分类 + 独立 OMI/NOMI 二分类，
不是四个独立 sigmoid 标签。

Baseline 五折已经完成，并已归档 `baseline/summary/summary.json`、阈值分析、
ROC/PR 曲线、混淆矩阵、训练/验证曲线以及五折脱敏 epoch 日志。LoRA fold0--3
已完成，fold4 正在 GPU1 运行；DoRA fold0 已在 GPU0 启动。因此 LoRA/DoRA 的
最终比较指标尚未生成。

脱敏日志只包含 phase、epoch、loss、准确率、学习率和最佳验证损失，不含原始
Fairseq JSONL、服务器路径、患者/记录标识、ECG 波形、checkpoint 或逐例预测。
