# ACS–OMI Joint-6 ECG 分类

本仓库保存 ACS–OMI ECG 研究中可公开版本化的代码、配置、聚合实验结果、脱敏训练曲线、可视化图和中文报告。原始 ECG、患者级表格、逐记录预测、预训练权重和训练 checkpoint 不进入 GitHub。

## 当前方案

当前主任务是单一 Joint-6 六分类，不是四个独立 sigmoid，也不是旧的 ACS/OMI 双 Head：

`OTHER / STEMI+nOMI / STEMI+OMI / NSTEMI+nOMI / NSTEMI+OMI / UA`

- 模型：ECG-LFM + LoRA + 6-way Softmax Head；
- 损失：CrossEntropyLoss；
- ACS 概率：`[q0, q1+q2, q3+q4, q5]`，使用 argmax；
- OMI 概率：`q2+q4`，inner-validation Youden 阈值冻结到 outer-test；
- 训练池：17,960 条 ECG、17,018 名患者；
- 划分：patient-level nested 5-fold，seed=2026。

旧 factorized two-head 实现和结果保留作消融与历史参照，但其 pooled OOF protocol 与当前 Joint-6 不完全相同，不能直接用于证明任务形式优劣。

## 当前结果

Joint-6 对比实验已完成 **7 个模型 × 5 folds = 35/35 folds**。当前最佳模型为 ECG-LFM LoRA：

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

## 主要实验结论

- ECG-LFM LoRA 在 Joint-6 Macro AUPRC、ACS Macro AUPRC、OMI AUPRC 和两个 NSTE-ACS 子组指标上均优于本次 Partial FT 运行与五种传统模型。
- 名为 Partial FT 的当前运行 provenance 实际显示 100% 参数可训练；LoRA 仅训练 0.939% 参数，峰值显存低约 25.7%。
- RDR 自适应重加权未优于未加权 W0；结论为 NO-GO，但不能泛化为所有不平衡学习方法均无效。
- PCGrad/CAGrad/FairGrad/Aligned-MTL 等多目标优化未通过 Stage-A gate。
- Direct RL、R1-v2 和 Cross-task RL 尚无可信涨点证据，不应作为已验证 contribution。
- Joint-7 与 Joint-10 均在训练前标签审计中停止，未通过的方案没有伪造标签或强行训练。
- 当前 Joint-6 尚无独立外部测试；历史平台结果不能冒充当前任务外部验证。

## 仓库结构

- `experiments/`：训练、nested split、Joint-6 评价和结果汇总；
- `configs/`：实验配置；
- `tests/`：标签映射、概率边缘化、梯度与 checkpoint 回归测试；
- `results/joint6_cv/`：当前 Joint-6 标签审计、每折聚合指标、脱敏曲线和汇总；
- `results/adaptive_weighting*`：W0–W6 与 RDR Fold0；
- `results/ecg_lfm_*`：多目标优化与 RL 探索；
- `docs/`：专项实验报告；
- `docs/reports/`：中文总报告与完整性审计；
- `docs/logs/`：脱敏日志索引。

## 安全与复现边界

禁止提交：

- SSH 密码、私钥或访问令牌；
- 原始 ECG/WFDB、患者标识或 patient-level assignments；
- 逐记录 OOF/测试预测、外部平台提交文件；
- 下载的模型权重和训练 checkpoint；
- 含患者信息、凭据或逐记录预测的原始日志；高频 fairseq event stream 不公开。

Joint-6 总调度日志保存在 `logs/joint6_cv/full_comparison_gpu1.log`，每折公开训练轨迹保存在 `training_curves.csv`；更高频的 fairseq event stream、checkpoint 和逐记录预测保留在受控服务器。数据边界见 [DATASET.md](DATASET.md)，仓库清单见 [REPOSITORY_MANIFEST.md](REPOSITORY_MANIFEST.md)。
