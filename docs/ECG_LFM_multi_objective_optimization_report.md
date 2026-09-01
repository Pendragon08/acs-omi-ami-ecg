# ECG-LFM 双任务多目标优化实验报告

> 生成时间：2026-09-01T23:45:48.541966+00:00。本报告只记录监督学习 MOO；RL 在本轮明确暂停。

## 1. 审计结论与任务定义

当前任务是联合双任务，而不是四个独立 sigmoid：ACS 头为互斥 `OTHER / STEMI / NSTEMI / UA` 四分类（4 个 logit、softmax、无权重 CrossEntropyLoss），OMI 头为独立二分类（1 个 logit、sigmoid、无权重 BCEWithLogitsLoss）。联合目标固定为 `L_ACS + L_OMI`。

本轮实际使用的模型是官方 ECG-LFM 编码器、1024-D pooled representation 和 `Linear(1024,5)` 输出头；仅 LoRA rank=8、alpha=16、dropout=0.05 参数可训练，骨干冻结。投影头前四行属于 ACS，最后一行属于 OMI；梯度 surgery 只作用于共享 LoRA 参数。

证据路径：`configs/ecg_lfm/corrected_joint_lora_moo.yaml`、`third_party/ECG-LFM/fairseq_signals/criterions/acs_joint_moo.py`、`third_party/ECG-LFM/fairseq_signals/models/classification/ecg_transformer_classifier.py`。

## 2. 数据、划分和公平性

训练使用既有 `manifests/ecg_lfm_nested`。每个 outer fold 的原有 patient-level 划分保持不变；outer-train 内部复用固定的 patient-level inner-train/inner-validation，`split_provenance.json` 保存 assignment hash 和零 patient overlap。Stage A 只使用 Fold0 的 inner-validation 做筛选和阈值选择，未读取 `outer_test.tsv`。

所有 MOO 方法共享同一 pretrained checkpoint、LoRA 配置、seed=2026、batch size=64、update_freq=4（effective batch 256）、Adam `(0.9,0.98)`、lr=1e-4、step scheduler(period=112, decay=0.99)、FP16、100 epoch 上限、patience=10、最小 inner-validation CE+BCE checkpoint criterion。

## 3. 现有基线 W0

W0 参考结果来自已有 nested evaluator 的 inner-validation JSON（不是外部测试）：

- ACS Macro AUROC=0.798900，Macro AUPRC=0.558565，Macro F1=0.531178，Balanced Accuracy=0.515910。
- NSTEMI AUPRC=0.414186，NSTEMI-vs-UA AUPRC=0.560098。
- OMI AUROC=0.918460，AUPRC=0.466715，F1=0.409037，Sensitivity=0.891192，Specificity=0.821589，MCC=0.427120，Youden threshold=0.040000。

证据：`results/multi_objective_evaluation/w0/fold0/inner_metrics.json`（若该文件缺失，则以 `results/adaptive_weighting_evaluation/w0/fold0/metrics.json` 的 inner_validation 部分为参考）。

## 4. MOO 实现与来源

新增 `acs_joint_moo` criterion 与 `multi_objective.py`。linear-sum 完全复现 W0；PCGrad、CAGrad(c=0.5)、FairGrad(alpha=1.0)、Aligned-MTL 使用公开仓库对应算法，且 solver fallback 会显式写入日志。Fairseq 原有 FP16 loss-scale、gradient accumulation、all-reduce、clipping 和 Adam step 未替换。

来源和 commit 记录见 `docs/logs/ECG_LFM_multi_objective_sources.md`。

## 5. Stage A：Fold0 inner-validation 筛选

筛选门槛：相对于 W0，`min(ΔACS Macro AUPRC, ΔOMI AUPRC) >= -0.003` 且 `max >= +0.005`，或两个 primary delta 都至少为 +0.003。该门槛只决定是否值得进入 Stage B，不构成泛化结论。

| 方法 | ACS Macro AUPRC | OMI AUPRC | NSTEMI AUPRC | NSTEMI-vs-UA AUPRC | OMI MCC | 冲突率 | Gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| linear_sum | 0.557115 | 0.451184 | 0.406807 | 0.558088 | 0.485381 | 0.285513 | FAIL |
| pcgrad | 0.555350 | 0.449083 | 0.404778 | 0.556416 | 0.492384 | 0.273071 | FAIL |
| cagrad | 0.554291 | 0.448739 | 0.409436 | 0.543548 | 0.458876 | 0.291007 | FAIL |
| fairgrad | 0.546761 | 0.440173 | 0.408177 | 0.551103 | 0.462851 | 0.283504 | FAIL |
| aligned_mtl | 0.552545 | 0.449032 | 0.400446 | 0.536102 | 0.471936 | 0.268916 | FAIL |

完整机器可读结果：`results/multi_objective_stage_a/stage_a_metrics.csv`、`stage_a_summary.json`。训练日志位于 `results/multi_objective/logs/`；诊断图位于 `results/multi_objective_stage_a/`。

## 6. 梯度冲突诊断

每个 train epoch 记录 shared ACS/OMI cosine、conflict flag、两任务梯度范数、合成梯度范数以及方法有效 task coefficient。冲突率定义为 batch 中 `cosine < 0` 的比例。头部梯度不参与 surgery，避免把 ACS/OMI task-specific row 错误混合。

应结合 `moo_conflict_rate.png`、`moo_gradient_cosine.png`、`moo_gradient_norms.png`、`moo_effective_task_weights.png` 判断冲突是否在 early/middle/late 阶段集中，以及诊断改善是否与验证指标同步；只有指标与冲突方向一致时，才能声称 MOO 解决了负迁移。

## 7. Stage B：完整 nested five-fold 状态

Stage B 尚未启动或尚无完整五折结果。根据报告协议，只有 Stage A gate 通过的方法才能进入 Stage B。


## 8. 校准、阈值和外部测试边界

ACS 继续使用四分类 softmax/argmax，不寻找互斥类别阈值；OMI threshold 仍只从 inner-validation 选择（0.5、max-F1、Youden、Sensitivity≥0.90、Sensitivity≥0.95、Specificity≥0.90），冻结后再用于 outer fold。MOO 不改变 calibration protocol。当前 MOO 阶段未提交 1,995 条 held-out external test，因此报告中的 outer-test 不能写成 external test。

## 9. 运行成本和可追溯性

每个方法的 `fairseq_train.jsonl` 和 `launch_provenance.json` 记录 config、checkpoint、split hash、源码 hash、GPU、时间和显存；服务器原始日志不进入 Git，去标识化摘要和图表进入仓库。当前训练使用 GPU1 单卡；GPU0 当时存在其他用户进程，未抢占。

## 10. 结论规则

- 若 Adaptive/MOO 在完整五折中 ACS Macro AUPRC 与 OMI AUPRC 均不劣于 W0，至少一个稳定提升，NSTEMI/NSTE OMI 不恶化且 calibration 稳定，才可记为 GO。
- 若只比 unweighted 好但与简单 linear-sum/其他静态策略相当，应记为“优化方向有价值”，不能声称某个 MOO 算法独立优越。
- 若 Fold0 通过但跨 fold 不稳定，或一个任务提升而另一任务下降，应记为 NO-GO。不能用 outer-test 反复调参。

### 当前状态

Stage A 没有方法通过预注册门槛，最终结论：NO-GO（不启动完整五折 MOO）。

## 11. 复现实验入口

- Stage A：`scripts/run_moo_stage_a.sh`（服务器 screen，GPU1）。
- Stage B：`scripts/run_moo_nested_5fold.sh METHOD ...`，仅允许传入已经通过 inner-validation gate 的方法。
- 单元测试：`scripts/run_moo_unit_tests.py`，服务器环境中 7 项已通过。
