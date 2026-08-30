# 实验日志索引与 GitHub 归档策略

更新时间：2026-08-30 08:49 UTC。下表的服务器路径用于内部审计；可公开的训练过程已
从原始日志中提取为不含路径、患者编号和逐例预测的
`training_log_sanitized.csv`，并随 `results/` 一起提交 GitHub。

## GitHub 中保存什么

- 每折/每 epoch 的训练损失、验证损失、验证指标和学习率（脱敏 CSV）；
- 损失与学习率曲线、ROC/PR 曲线、混淆矩阵；
- 汇总 JSON/CSV、阈值搜索、标签审计、外部平台聚合指标及 provenance。

原始 screen/Fairseq 日志、原始 ECG、患者级表格、checkpoint、逐例 OOF/外部
预测和提交 CSV 不上传。原始日志可能包含服务器绝对路径、数据定位信息或重复
的 Fairseq 事件流，仍留在受控服务器；报告引用本索引即可追溯到原始运行。

## 当前校正任务

| 运行 | GPU/状态 | 服务器原始日志 | GitHub 证据 |
|---|---|---|---|
| E4 HMoE/WavelNet/KAN 联合任务 | GPU1，已完成 | `logs/corrected/acs_joint_e4_gpu1.log` | `results/corrected_joint_acs_omi_e4/`、`docs/reports/corrected_joint_e4_20260829.md` |
| 传统架构对比 | GPU1，已完成 | `logs/corrected_joint_model_comparison/all_gpu1.log` | `results/corrected_joint_model_comparison/<模型>/`、`docs/reports/corrected_joint_model_comparison_20260829.md` |
| ECG-LFM Baseline/LoRA/DoRA 校正重跑 | GPU1/GPU0 并行（v2；Baseline/LoRA 五折完成并已有 LoRA pooled OOF 汇总，DoRA fold0--2 完成，fold3 在 GPU0 运行至 epoch 4，fold4 待运行） | `logs/corrected_ecg_lfm_joint_bs64_v2/master.log`、`dora_gpu0_master.log` 及各折日志 | `results/corrected_ecg_lfm_joint_bs64_v2/baseline/summary/`、`docs/reports/corrected_ecg_lfm_joint_20260829.md` |

## 已完成结果目录中的过程文件

- E4：`results/corrected_joint_acs_omi_e4/training_log_sanitized.csv`、
  `training_validation_loss_lr.png`、`roc_pr_curves.png`、`confusion_matrices.png`；
- InceptionTime、ResNet1D_Wang、LSTM、ViT、Transformer1D：每个模型目录均有
  `training_log_sanitized.csv`、上述三类图、`history_fold*.json`、汇总指标和
  外部平台 JSON；
- 生成脱敏日志的脚本：`scripts/export_sanitized_training_logs.py`。

## ECG-LFM 及历史日志

ECG-LFM 首次校正运行因 Fairseq 四列标签无法 reshape 到五个 logit 而停止，
该运行不产生有效指标；v2 已物化 `[OTHER,STEMI,NSTEMI,UA,OMI]` 标签并通过
fold0 smoke test。Baseline 已由评估器生成同格式汇总、图表和脱敏 epoch 日志；
LoRA/DoRA 完成后继续追加对应汇总。

历史损失/特征消融日志在服务器 `logs/loss_feature_ablation/`，报告中的
`docs/reports/loss_feature_ablation/EXPERIMENT_LOG.md` 是公开时间线；历史
独立 sigmoid 四标签模型的日志也只作为 `LEGACY/INVALID` 审计证据。

## 内部复核方法

通过批准的 SSH/SFTP 访问服务器时，只读取本索引给出的相对路径；不要把密码写
入脚本，也不要把原始日志复制进仓库。若需新增结果，请先运行脱敏导出脚本，
检查输出不含绝对路径、患者/记录标识和逐例概率，再提交汇总与图表。
