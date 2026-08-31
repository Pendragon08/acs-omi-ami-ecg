# E4 HMoE/WavelNet/KAN 联合任务结果

本目录保存 E4 的患者级五折汇总、标签审计、阈值分析、脱敏训练日志、损失/LR、
ROC/PR、混淆矩阵及外部平台聚合指标。Frequency Expert 使用端到端 WavelNet，
没有把 MVMD 作为网络外部预处理。

ACS 阳性亚型为 `STEMI/NSTEMI/UA` 单标签三分类；全队列内部保留 `OTHER`；
OMI 为独立二分类。外部结果见 `official_platform_metrics.json`。
