# 实验日志索引与 GitHub 归档策略

**更新时间：** 2026-08-31

## GitHub 保存内容

- 每折/每 epoch 的训练损失、验证损失、验证指标和学习率脱敏 CSV；
- 损失与学习率曲线、ROC/PR 曲线、混淆矩阵；
- 汇总 JSON/CSV、阈值搜索、标签审计、外部平台聚合指标和 provenance；
- 中文实验报告与完整性审计。

原始 ECG、患者级表格、checkpoint、逐例 OOF/外部预测、提交 CSV、下载权重及含
服务器绝对路径的原始日志不进入 GitHub。原始运行日志保留在受控服务器；GitHub 中的
`training_log_sanitized.csv` 仅保留 fold、epoch、loss、验证指标和学习率。

## 当前有效实验

| 实验 | 运行状态 | 受控服务器日志 | GitHub 证据 |
|---|---|---|---|
| E4 HMoE/WavelNet/KAN 联合任务 | GPU1，已完成 | `logs/corrected/acs_joint_e4_gpu1.log` | `results/corrected_joint_acs_omi_e4/` |
| 传统架构对比 | GPU1，已完成 | `logs/corrected_joint_model_comparison/all_gpu1.log` | `results/corrected_joint_model_comparison/` |
| ECG-LFM Baseline/LoRA/DoRA | 已完成；恢复运行均绑定单卡 | `logs/corrected_ecg_lfm_joint_bs64_v2/` 下的 master/折日志 | `results/corrected_ecg_lfm_joint_bs64_v2/` |
| 历史 OMI/AMI 损失与特征消融 | 已完成 | `logs/loss_feature_ablation/` | `results/loss_feature_ablation/` |

统一结果解释见 `docs/reports/实验结果总报告_最终.md`，完整性审计见
`docs/reports/实验完整性审计_20260831.md`。

## 结果目录中的过程文件

- E4：`training_log_sanitized.csv`、`training_validation_loss_lr.png`、
  `roc_pr_curves.png`、`confusion_matrices.png`；
- 五种传统模型：各模型目录保存脱敏训练日志、三类图、`history_fold*.json`、汇总指标和
  `official_platform_metrics.json`；
- ECG-LFM：Baseline/LoRA/DoRA 各保存五折汇总、脱敏日志、损失/LR、ROC/PR、
  混淆矩阵和平台聚合 JSON；
- 历史消融：五种子汇总、阈值策略、bootstrap、脱敏日志与曲线。

生成脱敏训练日志的脚本为 `scripts/export_sanitized_training_logs.py`。

## 审核与复现规则

1. 服务器长任务必须使用 `screen`，写日志，并通过 `CUDA_VISIBLE_DEVICES` 绑定一张 GPU。
2. 新结果提交前先检查日志不含密码、绝对数据路径、患者标识和逐例概率。
3. 报告中的数值必须能追溯到当前 `results/corrected_*` 或明确标为历史消融的目录。
4. 旧 STEMI/NSTEMI/UA 三个独立 sigmoid 的错误任务已从当前分支删除，只能从 Git 历史
   恢复，不能作为当前任务的实验依据。
5. 外部平台已经参与模型选择，当前平台指标只能作为描述性工程证据。
