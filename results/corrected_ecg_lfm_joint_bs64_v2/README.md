# ECG-LFM 联合任务结果

本目录保存 Baseline、LoRA、DoRA 的五折 pooled OOF 汇总、脱敏逐 epoch 日志、
训练/验证损失和学习率曲线、ROC/PR、混淆矩阵及外部平台聚合指标。

ACS 阳性亚型使用 `STEMI/NSTEMI/UA` 单标签三分类；全队列训练为了保留
non-ACS，内部增加 `OTHER` 状态。OMI 是独立二分类，只有 OMI 使用验证集阈值。

- `baseline/`：ECG-LFM 基线微调；
- `lora/`：rank=8、alpha=16 的 LoRA；
- `dora/`：rank=8、alpha=16 的 DoRA；
- `summary/summary.json`：五折 pooled OOF 汇总；
- `official_platform_metrics.json`：外部平台 hard-label 指标。

最终对比和训练成本见 `docs/reports/实验结果总报告_最终.md`。原始日志和模型
权重保留在受控服务器，不进入本仓库。
