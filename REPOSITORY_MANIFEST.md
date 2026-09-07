# GitHub 归档清单

_更新于 2026-09-07 · 按用途建立索引，保持代码和实验产物原路径_

---

## 📦 当前内容与位置

| 类别 | 位置 | 内容 |
|---|---|---|
| 当前实验代码 | `experiments/`、`models/`、`nets/` | 联合任务训练、推理和模型适配 |
| ECG-LFM 上游代码 | `third_party/ECG-LFM/` | 本实验使用的代码快照，不含权重 |
| 配置 | `configs/` | 当前联合实验与历史消融配置；当前入口为 `configs/ecg_lfm/corrected_joint6_lora.yaml` |
| 报告导航 | [docs/README.md](docs/README.md) | 当前、在研、历史三类中文入口 |
| 总报告 | [docs/reports/](docs/reports/README.md) | 中文总报告、审计和历史消融报告 |
| 日志索引 | `docs/logs/` | 服务器原始日志与脱敏产物的映射 |
| 已完成主结果 | [results/joint6_cv/](results/joint6_cv/README.md) | 7 个模型、35/35 folds；指标、曲线、图和修正凭据 |
| 在研优化 | [results/joint6_optimization_20260907/](results/joint6_optimization_20260907/README.md) | 报告修复完成；Fold0 学习率对比尚未完成 |
| 历史结果 | [results/README.md](results/README.md) | corrected 双 Head、RDR、MOO、RL、旧损失/特征消融 |
| 测试 | `tests/` | 标签、边缘概率、checkpoint、评价、报告和筛选规则测试 |

## 📚 历史保留与已弃用内容

此前清理已移除或替代的错误/重复资产包括：

- STEMI/NSTEMI/UA/OMI 四个独立 sigmoid 的训练、阈值和提交代码；
- 旧 ECG-LFM 四独立输出、旧 HMoE 多标签和旧传统模型结果目录；
- 已被当前流程替代的恢复脚本、重复配置、过程快照和中英文混杂报告；
- 使用 MVMD 作为网络外部预处理的旧辅助脚本。

这些内容仍存在于 Git 历史中，但不能作为当前性能证据。

本轮没有再次删除实验、日志、模型实现或改写 Git 历史。双 Head、Joint-7/10 标签审计、RDR/MOO/RL 等保留原路径并标记历史。报告修复前的 10 折曲线放在各折 `legacy_reporting_v1/`，仅为可追溯备份，不参与当前分析。

## 🔐 禁止提交

原始 ECG、患者表格、split 索引、逐例预测、提交 CSV、checkpoint、下载权重、
未经审阅的原始训练日志和任何凭据都禁止提交。每次推送前必须检查 staged 文件和
`.gitignore`，确认没有受控数据或敏感信息。

`joint10_label_audit/raw_coronary_fields.csv` 是血管字段释义表，不是逐患者原始数据；`patient_joint_counts.csv` / `joint10_patient_counts.csv` 是类别患者数汇总，不含患者 ID。保留这些文件不等于允许提交患者级表格。既有已跟踪日志的说明见[日志索引](docs/logs/EXPERIMENT_LOG_INDEX.md)。

## 🔍 可追溯要求

每个有效实验应记录任务定义、数据/划分、seed、模型与损失、阈值规则、主要
指标、配置路径和结果目录。内部 pooled OOF 同时承担 checkpoint/阈值选择时，
必须明确标注为描述性结果；外部平台未返回概率时不得虚构 AUROC/AUPRC。

发布流程见[协作说明](CONTRIBUTING.md)，本轮变更见[仓库整理记录](docs/reports/仓库整理记录_20260907.md)。GitHub 报告是发布时快照，不能代替运行服务器的实时状态。
