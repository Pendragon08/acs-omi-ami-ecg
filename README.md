# ACS–OMI Joint-6 ECG 分类

_中文研究归档 · 更新于 2026-09-07 · 已完成主结果与在研优化分开记录_

---

本仓库保存 ACS–OMI ECG 研究中可公开版本化的代码、配置、聚合实验结果、脱敏训练曲线、可视化图和中文报告。原始 ECG、患者级表格、逐记录预测、预训练权重和训练 checkpoint 不进入 GitHub。

## 📍 从哪里开始

| 想查看的内容 | 入口 |
|---|---|
| 当前模型、已完成对比与分析 | [中文实验结果总报告](docs/reports/实验结果总报告_最终.md) |
| 本轮曲线/成本修复及学习率优化 | [分阶段优化报告](docs/ECG_LFM_Joint6_Optimization_Report.md) |
| 所有当前与历史报告 | [文档导航](docs/README.md) |
| 交给 Codex 在 Overleaf 写论文 | [论文写作交接说明](docs/PAPER_WRITING_HANDOFF.md) |
| 指标、图表、训练轨迹 | [结果导航](results/README.md) · [日志索引](docs/logs/EXPERIMENT_LOG_INDEX.md) |
| 协作与安全发布 | [协作说明](CONTRIBUTING.md) · [仓库清单](REPOSITORY_MANIFEST.md) |

## 🎯 当前方案

当前主任务是单一 Joint-6 六分类，不是四个独立 sigmoid，也不是旧的 ACS/OMI 双 Head：

`OTHER / STEMI+nOMI / STEMI+OMI / NSTEMI+nOMI / NSTEMI+OMI / UA`

- 模型：ECG-LFM + LoRA + 6-way Softmax Head；
- 损失：CrossEntropyLoss；
- ACS 概率：`[q0, q1+q2, q3+q4, q5]`，使用 argmax；
- OMI 概率：`q2+q4`，inner-validation Youden 阈值冻结到 outer-test；
- 训练池：17,960 条 ECG、17,018 名患者；
- 划分：patient-level nested 5-fold，seed=2026。

旧 factorized two-head 实现和结果保留作消融与历史参照，但其 pooled OOF protocol 与当前 Joint-6 不完全相同，不能直接用于证明任务形式优劣。

## 📊 已完成的主结果

Joint-6 对比实验已完成 **7 个模型 × 5 folds = 35/35 folds**。按本次对比的主要指标 Joint-6 Macro AUPRC，最佳模型为 ECG-LFM LoRA；不表示每个单项指标均为最高：

| 指标 | 五折 mean ± std |
|---|---:|
| Joint-6 Macro AUPRC | **0.3843 ± 0.0056** |
| Joint-6 Macro AUROC | 0.8122 ± 0.0082 |
| Joint-6 Macro F1 | **0.3557 ± 0.0095** |
| ACS Macro AUPRC | **0.5445 ± 0.0067** |
| OMI AUROC | 0.9104 ± 0.0111 |
| OMI AUPRC | **0.4299 ± 0.0258** |
| OMI Sensitivity | 0.8256 ± 0.0608 |
| OMI Specificity | 0.8446 ± 0.0195 |
| NSTEMI vs UA AUPRC | **0.5527 ± 0.0363** |
| OMI among NSTE-ACS AUPRC | **0.2695 ± 0.0599** |

最难的联合类别是 `NSTEMI+OMI`：AUPRC 0.1615 ± 0.0426，Sensitivity 0.0536 ± 0.0373。

完整表格、逐类指标、训练成本和结果分析见：

- [中文实验结果总报告](docs/reports/实验结果总报告_最终.md)
- [Joint-6 五折模型对比报告](docs/ECG_LFM_Joint6_CV_Comparison_Report.md)
- [Joint-6 机器可读汇总](results/joint6_cv/summary/)
- [实验日志索引](docs/logs/EXPERIMENT_LOG_INDEX.md)

## 🔄 在研优化与本次修复

2026-09-07 11:54（北京时间）核验：GPU0 的 `lr5e5` 与 GPU1 的 `lr3e5` 仍在训练；仅运行原 nested Fold0 的学习率对比，尚无两组完整结果。后续静态加权、校准、追加五折与多 seed 未启动。GitHub 是发布时快照，不是服务器实时看板。

已完成 10 个 ECG-LFM 折的训练曲线修复、有效训练参数统计修正和历史 evaluator 版本兼容检查，38 项测试通过，35/35 历史结果完整性复核通过；**原始预测与分类指标未改写**。旧曲线保存在各折 `legacy_reporting_v1/`，不能再用作当前分析。见[优化报告](docs/ECG_LFM_Joint6_Optimization_Report.md)及[测试凭据](results/joint6_optimization_20260907/verification_summary.json)。

## 🔍 主要实验结论

- ECG-LFM LoRA 在 Joint-6 Macro AUPRC、ACS Macro AUPRC、OMI AUPRC 和两个 NSTE-ACS 子组指标上均优于本次 Partial FT 运行与五种传统模型。
- LoRA 有效训练参数占 0.939%；本次 Partial FT 虽然 `requires_grad` 为 100%，但卷积前端受 `no_grad` 冻结，有效参与更新的参数占 **99.936%**。不可将其简单写成全参微调，也不是仅训练少量层的轻量 Partial FT。历史显存/耗时测量范围不统一，不能直接作为严格训练成本提升证据，见[修正成本表](results/joint6_cv/summary/training_cost.csv)。
- RDR 自适应重加权未优于未加权 W0；结论为 NO-GO，但不能泛化为所有不平衡学习方法均无效。
- PCGrad/CAGrad/FairGrad/Aligned-MTL 等多目标优化未通过 Stage-A gate。
- Direct RL、R1-v2 和 Cross-task RL 尚无可信涨点证据，不应作为已验证 contribution。
- Joint-7 与 Joint-10 均在训练前标签审计中停止，未通过的方案没有伪造标签或强行训练。
- 当前 Joint-6 尚无独立外部测试；历史平台结果不能冒充当前任务外部验证。

## 📦 仓库结构

- `experiments/`：训练、nested split、Joint-6 评价和结果汇总；
- `configs/`：实验配置；
- `tests/`：标签映射、概率边缘化、梯度与 checkpoint 回归测试；
- `results/joint6_cv/`：当前 Joint-6 标签审计、每折聚合指标、脱敏曲线和汇总；
- `results/joint6_optimization_20260907/`：当前 Fold0 优化的安全聚合产物；
- `results/adaptive_weighting*`：W0–W6 与 RDR Fold0；
- `results/ecg_lfm_*`：多目标优化与 RL 探索；
- `docs/`：专项实验报告；
- `docs/reports/`：中文总报告与完整性审计；
- `docs/logs/`：脱敏日志索引。

## 🔐 安全与复现边界

禁止提交：

- SSH 密码、私钥或访问令牌；
- 原始 ECG/WFDB、患者标识或 patient-level assignments；
- 逐记录 OOF/测试预测、外部平台提交文件；
- 下载的模型权重和训练 checkpoint；
- 含患者信息、凭据或逐记录预测的原始日志；新增高频 fairseq event stream 不直接公开。

已归档的 [Joint-6 总调度日志](logs/joint6_cv/full_comparison_gpu1.log)与少量历史 RL 聚合 JSONL 保留，不代表允许直接上传任何原始日志。每折公开训练轨迹以修正后的 `training_curves.csv` 为准；本轮优化原始日志、checkpoint 和逐记录预测保留在受控服务器。数据边界见 [DATASET.md](DATASET.md)。
