# Overleaf 论文写作交接说明

_更新于 2026-09-07 · 供从 GitHub 读取本项目的 Codex 使用_

---

## 📍 先固定证据版本

开始写论文前记录实际使用的 Git commit SHA。代码、配置、结果与报告必须来自同一版本；服务器在研实验不等于本提交已有最终结果。优先阅读[中文总报告](reports/实验结果总报告_最终.md)、[五折对比报告](ECG_LFM_Joint6_CV_Comparison_Report.md)、[评价报告](ECG_LFM_Joint6_Evaluation_Report.md)及[本轮审计修订](ECG_LFM_Joint6_Optimization_Report.md)。

## 🎯 当前方法与数据

当前方法是 **ECG-LFM + LoRA + Joint-6 单一 Softmax Head + CrossEntropyLoss**，不是旧 ACS 四类 CE + OMI BCE 的 factorized two-head，也不是六个独立 sigmoid。

六类为 OTHER、STEMI+nOMI、STEMI+OMI、NSTEMI+nOMI、NSTEMI+OMI、UA。
ACS 概率为 `[q0, q1+q2, q3+q4, q5]`，prediction 使用 argmax；OMI 概率为 `q2+q4`。OMI Youden 阈值在每折 inner-validation 上选择并冻结到 outer-test。

训练池 17,960 条 ECG、17,018 名患者；另有 1,995 条外部盲测记录。patient-disjoint nested 5-fold，seed=2026。评价指标以 ECG record 为单位，不是患者概率聚合。唯一 OTHER+OMI 例外保留 raw OMI 标签；不删除或篡改标签。详细说明及来源路径见 [DATASET.md](../DATASET.md)与[标签审计](../results/joint6_cv/label_audit/)。

实际训练设置读取[LoRA YAML](../configs/ecg_lfm/corrected_joint6_lora.yaml)与[训练入口](../experiments/ecg_lfm/run_joint6_fold.py)。LoRA r=8、alpha=16、dropout=0.05；原五折基线 LR=1e-4、batch=64、update_freq=4、Adam、step scheduler、max_epoch=100、early-stopping patience=10。不是所有折都训练 100 epochs；以各折完成记录为准。checkpoint 按最低 inner-validation Joint CE 选择。

## 📊 论文结果与图表来源

| 内容 | 证据 |
|---|---|
| 七种模型 × 五折完成度 | [completion_status.json](../results/joint6_cv/summary/completion_status.json) |
| 五折 mean/std 与 CI | [summary_metrics.csv](../results/joint6_cv/summary/summary_metrics.csv) |
| 每折完整指标 | [fold_metrics.csv](../results/joint6_cv/summary/fold_metrics.csv) |
| 六类指标 | [per_class_summary.csv](../results/joint6_cv/summary/per_class_summary.csv) · [per_class_metrics.csv](../results/joint6_cv/summary/per_class_metrics.csv) |
| pooled OOF 聚合指标 | [pooled_oof_metrics.json](../results/joint6_cv/summary/pooled_oof_metrics.json) |
| 参数与训练成本 | [training_cost.csv](../results/joint6_cv/summary/training_cost.csv) · [corrected_training_statistics.csv](../results/joint6_cv/summary/corrected_training_statistics.csv) |
| ECG-LFM 图表与曲线 | [LoRA](../results/joint6_cv/evaluation/lora/) · [Partial FT](../results/joint6_cv/evaluation/partial/) |
| 传统模型图表与曲线 | [traditional/](../results/joint6_cv/traditional/) |

当前已完成模型对比中，按 Joint-6 Macro AUPRC 最佳的是 LoRA：0.3843 ± 0.0056；ACS Macro AUPRC=0.5445 ± 0.0067，OMI AUPRC=0.4299 ± 0.0258。它不代表 OMI Sensitivity 等每个单项都最高。最难联合类 NSTEMI+OMI 的 AUPRC=0.1615 ± 0.0426、argmax Sensitivity=0.0536 ± 0.0373，不能隐藏。

## 🔍 必须纳入论文的审计修订

1. 修正前曲线曾混入其他折、重复 epoch 及错误重跑片段；只使用当前各折 `training_curves.csv` 与对应 PNG，不使用 `legacy_reporting_v1/`。
2. Partial FT 的 `requires_grad=100%` 不是实际更新比例：卷积前端 `no_grad` 后，有效训练参数为 622,481,158 / 622,882,054（99.936%）。LoRA 为 5,904,390 / 628,780,294（0.939%）。
3. 历史 ECG-LFM 显存是整卡 30 秒采样，传统模型是 PyTorch allocated 峰值；耗时范围也不同。不能直接把旧百分比差异写成严格成本优势。
4. 原 metrics、预测和 receipt 未修改；`reporting_corrections.json` 与精确版本兼容声明记录了修订，38 项测试与 35/35 历史结果复核有凭据。

## ⚠️ 不得超出的结论边界

- 当前没有 Joint-6 独立外部测试指标，不得以历史平台结果补充。
- 五折不是五个 seed；fold-mean 的 Student-t 95% CI 不是 patient-level paired bootstrap。
- 传统模型用 P2，ECG-LFM 用原生预处理，本次是工程方案对比，不是单一网络结构消融。
- 历史双 Head 与 Joint-6 的 checkpoint/threshold 协议不同，不能宣称已证明联合标签建模优于 factorized。
- RDR/MOO/RL 结果来自历史任务或探索阶段，不是当前 Joint-6 的组件，也不是已确认有效 contribution。
- 两个新 LR 候选在发布核验时尚未完成；不能把基线 inner-validation AUPRC=0.394075 当作 outer 五折成绩。
- 原始波形、患者表、checkpoint 和逐例预测未公开；不得声称可仅凭 GitHub 独立重新核对所有患者级指标。

## ✍️ 给论文写作 Codex 的执行要求

先建立“论文主张 → 配置/代码 → 结果文件”的对应清单，再在 Overleaf 编写 Methods、Results 与 Discussion。数值以同提交的机器可读结果为准；报告冲突必须查证，缺失证据标记待补，不猜测。参考论文的来源与引文另行核验，不能凭本项目说明虚构文献。此次交接不授权新增训练、外部测试提交或改变模型配置。
