# GitHub 归档清单

## 保留内容

| 类别 | 位置 | 内容 |
|---|---|---|
| 当前实验代码 | `experiments/`、`models/`、`nets/` | 联合任务训练、推理和模型适配 |
| ECG-LFM 上游代码 | `third_party/ECG-LFM/` | 本实验使用的代码快照，不含权重 |
| 配置 | `configs/` | 当前联合实验与有效消融配置 |
| 报告 | `docs/reports/` | 中文总报告、审计和有效消融报告 |
| 日志索引 | `docs/logs/` | 服务器原始日志与脱敏产物的映射 |
| 聚合结果 | `results/corrected_*`、`results/loss_feature_ablation/` | 汇总指标、脱敏日志和图表 |

## 已清理内容

当前分支不再保留以下错误或重复资产：

- STEMI/NSTEMI/UA/OMI 四个独立 sigmoid 的训练、阈值和提交代码；
- 旧 ECG-LFM 四独立输出、旧 HMoE 多标签和旧传统模型结果目录；
- 已被当前流程替代的恢复脚本、重复配置、过程快照和中英文混杂报告；
- 使用 MVMD 作为网络外部预处理的旧辅助脚本。

这些内容仍存在于 Git 历史中，但不能作为当前性能证据。

## 禁止提交

原始 ECG、患者表格、split 索引、逐例预测、提交 CSV、checkpoint、下载权重、
原始训练日志和任何凭据都禁止提交。每次推送前必须检查 staged 文件和
`.gitignore`，确认没有受控数据或敏感信息。

## 可追溯要求

每个有效实验应记录任务定义、数据/划分、seed、模型与损失、阈值规则、主要
指标、配置路径和结果目录。内部 pooled OOF 同时承担 checkpoint/阈值选择时，
必须明确标注为描述性结果；外部平台未返回概率时不得虚构 AUROC/AUPRC。
